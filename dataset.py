import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder

from utils import resize_image, flip_image, normalize


class OAIData:
    def __init__(self, config, phase, views, transform=None):
        self.config = config
        self.phase = phase
        self.image_views = [p for p in views if p != 'tab']
        self.transform = transform

        self.side_dict = {'RIGHT': 1, 'LEFT': 2}
        self.month_dict = {'0': '00', '12': '01', '24': '03', '36': '05', '48': '06', '60': '07', '72': '08', '96': '10'}
        self.zero_img = np.zeros((self.config['img_size'], self.config['img_size']))

        self.sheet_path = os.path.join(config['data_sheet_folder'], phase + '_data.csv')
        self.df_label = pd.read_csv(os.path.join(config['data_sheet_folder'], config['label_sheet_name']), sep=',')

        # KLG pred do not have label at the 60th month
        self.available_month = ['0', '12', '24', '36', '48', '72'] if 'V00WOMKP' in self.df_label.keys() else ['0', '12', '24', '48', '72']

        # prepare for tabular data
        self.enrollee = dict()
        df_tab = pd.read_csv(os.path.join(config['data_sheet_folder'], config['tabular_sheet_name']), sep=',')
        self.enrollee['data'], self.enrollee['mask'], self.enrollee['cat_idxs'], self.enrollee['cat_dims'], \
            self.enrollee['con_idxs'] = self.preprocess_tab(df_tab)

        # prepare for image data
        self.images, self.image_names = dict(), dict()
        for p in self.image_views:
            npzfile = np.load(os.path.join(config['image_folder'], p + '.npz'))
            self.images[p] = npzfile['x']
            self.image_names[p] = list(npzfile['y'])

        self.info_dic = {}    # save all image names (keys) and img path (items) and tabular (items)
        self.name_list = []  # save all image name
        self.image_list = []    # save all the images that are loaded into memory

        self.get_data()

    def preprocess_tab(self, df):
        df = df.replace('.: Missing Form/Incomplete Workbook', np.nan)
        df_mask = df.copy()  # all missing data in the mask are nan

        types = df.dtypes
        before_ID = True        # if before_ID, column idx is the true idx, else idx has to -1
        cat_idxs, cat_dims, con_idxs = [], [], []
        for i in range(len(df.columns)):
            col = df.columns[i]
            if col == 'ID':
                before_ID = False
                continue
            idx = i if before_ID else i - 1
            if types[col] == 'object':
                df[col] = df[col].fillna("Missing")
                l_enc = LabelEncoder()
                df[col] = l_enc.fit_transform(df[col].values)
                cat_idxs.append(idx)
                cat_dims.append(len(l_enc.classes_))
            else:
                df[col].fillna(-1, inplace=True)
                con_idxs.append(idx)
        return df, df_mask, cat_idxs, cat_dims, con_idxs

    def get_image(self, view, name):
        idx = self.image_names[view].index(name) if name in self.image_names[view] else -1
        if idx >= 0:        # image exist
            img = self.images[view][idx]
            img = self.preprocess_thickness(img) if 'thickness' in view else \
                self.preprocess_img(img, flip=(view == 'knee' and self.config['flip'] and 'RIGHT' in name))
            mask = 1
        else:
            img = self.zero_img
            mask = 0
        return img, mask

    def get_tab(self, patient_id, month):
        tab = self.enrollee['data'].loc[(self.enrollee['data'].ID == patient_id + '_' + month)]
        mask = self.enrollee['mask'].loc[(self.enrollee['mask'].ID == patient_id + '_' + month)]
        if len(tab) == 0:
            return None
        tab = tab.drop(columns=['ID'])
        mask = mask.drop(columns=['ID'])
        mask = (mask.notnull()).astype('int')
        tab = tab.values[0]
        mask = mask.values[0]

        cat_idxs = self.enrollee['cat_idxs']
        con_idxs = self.enrollee['con_idxs']
        cat_tab = tab[cat_idxs].copy().astype(np.int64)  # categorical columns
        con_tab = tab[con_idxs].copy().astype(np.float32)  # numerical columns
        cat_mask = mask[cat_idxs].copy().astype(np.int64)  # categorical columns
        con_mask = mask[con_idxs].copy().astype(np.int64)  # numerical columns
        return cat_tab, con_tab, cat_mask, con_mask

    def add_data(self, patient_id, month):
        cat_tab, con_tab, cat_mask, con_mask = self.get_tab(patient_id, month)
        side = list(self.side_dict.keys())
        for i in range(len(side)):
            name = patient_id + '_' + side[i]
            if name not in self.info_dic.keys():
                self.info_dic[name] = dict()
                self.name_list.append(name)
            self.info_dic[name][month] = dict(
                tab=[np.append(cat_tab, self.side_dict[side[i]]-1), con_tab],       # add side info to tabular data
                tab_mask=[np.append(cat_mask, 1), con_mask],
                label=self.get_label(name + '_' + month)
            )
            if len(self.image_views) > 0:
                mask = []
                for i in range(len(self.image_views)):
                    n = name + '_' + month if ('knee' in self.image_views[i] or 'thickness' in self.image_views[i]) else patient_id + '_' + month
                    img, m = self.get_image(self.image_views[i], n)
                    self.info_dic[name][month][self.image_views[i]] = img
                    mask.append(m)
                self.info_dic[name][month]['img_mask'] = mask

    def get_data(self):
        df = pd.read_csv(self.sheet_path, sep=',')
        all_patients = np.unique(df['patient_id'])
        for patient in all_patients:
            if len(self.name_list) >= self.config['num_load'] > 0:  # early return, used when debug
                return
            for month in self.available_month:
                self.add_data(str(patient), month)

    def get_label(self, name):
        patient_id, side, month = name.split('_')
        pred_month = str(int(month) + 24)       # pred the next 24 month result
        row = self.df_label.loc[(self.df_label.ID == int(patient_id)) & (self.df_label.SIDE == self.side_dict[side])]
        if len(row) == 0:  # row not found
            return None
        h = 'WOMKP' if 'V00'+'WOMKP' in row.keys() else 'XRKL'
        label = np.nanmax(getattr(row, 'V'+self.month_dict[pred_month]+h).values)  # in case multiple rows
        if np.isnan(label):  # no label
            return None
        return label

    def preprocess_img(self, img, flip=False):
        img = resize_image(img.astype(np.float32), target_size=[self.config['img_size'], self.config['img_size']])
        img = normalize(img, percentage_clip=99, zero_centered=False)
        if flip:
            img = flip_image(img)
        return img

    def preprocess_thickness(self, img):
        img = resize_image(img.astype(np.float32), target_size=[self.config['img_size'], self.config['img_size']])
        img[np.isnan(img)] = 0
        return img

    def get_tab_info(self):
        return [np.append(self.enrollee['cat_dims'], 2), len(self.enrollee['con_idxs'])]

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        name = self.name_list[idx]
        sample = {'name': name, 'label': dict(), 'tab': dict(), 'tab_mask': dict()}

        for month in self.available_month:
            if self.info_dic[name][month]['label'] is None:
                sample['label'][month] = -1
            else:
                sample['label'][month] = self.info_dic[name][month]['label']
            sample['tab'][month] = self.info_dic[name][month]['tab']
            sample['tab_mask'][month] = self.info_dic[name][month]['tab_mask']

        if len(self.image_views) > 0:  # has image
            sample['img_mask'] = dict()
            for p in self.image_views:
                sample[p] = dict()

            for month in self.available_month:
                for i in range(len(self.image_views)):
                    sample[self.image_views[i]][month] = self.info_dic[name][month][self.image_views[i]]
                    sample['img_mask'][month] = self.info_dic[name][month]['img_mask']
                    if self.transform is not None:
                        sample[self.image_views[i]][month] = self.transform(sample[self.image_views[i]][month])
        return sample


class ToTensor(object):
    def __call__(self, sample):
        n_tensor = torch.from_numpy(sample.copy())
        return n_tensor


def get_loader(config, phase, shuffle, drop_last=False):
    transform = transforms.Compose([ToTensor()])
    dataset = OAIData(config['data'], phase=phase, views=config['views'], transform=transform)
    tab_info = dataset.get_tab_info()
    loader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=shuffle, drop_last=drop_last)
    print("finish loading {} data".format(len(dataset)))
    return loader, tab_info


