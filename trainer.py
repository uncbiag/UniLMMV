import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.nn.utils.clip_grad import clip_grad_norm_
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score, accuracy_score, average_precision_score

from utils import get_WOMAC_label, get_KLG_label, aug_img


class Trainer():
    def __init__(self, config, net, logger, result_types):
        self.config = config
        self.net = net
        self.logger = logger
        self.result_types = result_types

        self.ckp_folder = os.path.join(config['save_folder'], 'checkpoint')
        os.makedirs(self.ckp_folder, exist_ok=True)
        tensorboard_folder = os.path.join(config['save_folder'], 'tensorboard')
        os.makedirs(tensorboard_folder, exist_ok=True)
        self.writer = SummaryWriter(tensorboard_folder)

        self.best = 0
        self.loss_types = ['cls']
        self.loss_dict, self.result_dict = dict(), dict()

    def clear_loss(self):
        for t in self.loss_types:
            self.loss_dict[t] = 0

    def clear_result(self):
        for t in self.result_types:
            self.result_dict[t] = 0

    def set_zero_grad(self):
        self.net.module.zero_grad()

    def set_train(self):
        self.net.module.train()

    def set_eval(self):
        self.net.module.eval()

    def save_networks(self, save_path):
        torch.save(self.net.state_dict(), save_path)

    def prepare_data(self, data, augmentation):
        months = list(data['label'].keys())
        views = self.config['views'].copy()

        if 'WOMAC' in self.config['data']['label_sheet_name']:
            score = get_WOMAC_label(data['label'])
        else:
            score = get_KLG_label(data['label'])
        label = []
        for m in months:
            label.append(score[m].type(torch.LongTensor).cuda())

        if 'tab' in views:      # tab in views
            tab, tab_mask = [], []
            for m in months:
                t = data['tab'][m]
                tm = data['tab_mask'][m]
                tab.append([t[i].cuda() for i in range(len(t))])      # [cat, cont]
                tab_mask.append([tm[i].cuda() for i in range(len(tm))])
            views.remove('tab')
        else:
            tab = tab_mask = None

        if len(views) > 0:      # image in views
            image, image_mask = [[] for _ in range(len(views))], [[] for _ in range(len(views))]
            zero_img = torch.zeros_like(data[views[0]]['0'][0]).to(torch.float)
            for i in range(len(views)):
                for m in months:
                    if 'knee' in views[i] or 'thickness' in views[i] or (m == '0' or m == '48'):       # (is knee) or (is thickness) or (is 0 or 48m)
                        img = data[views[i]][m].to(torch.float)
                        if augmentation:
                            contract_adjust = False if 'thickness' in views[i] else True
                            img = aug_img(img, zero_img, contract_adjust)
                        image[i].append(img.unsqueeze(1).cuda())
                        image_mask[i].append(data['img_mask'][m][i])
                    else:
                        image[i].append(None)
                        image_mask[i].append(torch.zeros_like(data['img_mask']['0'][0]))
        else:
            image = image_mask = None
        return image, image_mask, tab, tab_mask, label

    def print_loss_result(self, phase, prefix, length, is_train, print_result=True):
        # print loss
        if len(self.loss_dict.keys()) > 0:
            for key in list(self.loss_dict.keys()):
                self.loss_dict[key] /= length
            losses = ''
            for key in self.loss_dict:
                losses += ('%s: %.2f | ' % (key, self.loss_dict[key].data.item()))
                if is_train:
                    self.writer.add_scalar('%s/loss_%s' % (phase, key), self.loss_dict[key].data.item(), self.epoch)
            output = '{} \n\t{}'.format(prefix, losses)
        else:
            output = ''

        # print result
        if print_result:
            result = ''
            for key in self.result_dict:
                result += ('%s: %.4f | ' % (key, self.result_dict[key]))
                if is_train:
                    self.writer.add_scalar('%s/%s' % (phase, key), self.result_dict[key], self.epoch)
            output = "{}\n\t{}".format(output, result)

        self.logger.info(output)

    def get_result(self, pred, label):
        pred = np.array(pred)
        label = np.array(label)
        if 'roc' in self.result_types:
            roc = roc_auc_score(y_true=label, y_score=pred[:, 1]) if self.config['attention']['output_dim'] == 2 \
                else roc_auc_score(y_true=label, y_score=pred, multi_class='ovr')
            self.result_dict['roc'] += roc
        if 'ap' in self.result_types:
            ap = average_precision_score(y_true=label, y_score=pred[:, 1]) if self.config['attention']['output_dim'] == 2 \
                else average_precision_score(y_true=label, y_score=pred)
            self.result_dict['ap'] += ap
        pred = np.argmax(pred, axis=1)
        if 'f1score' in self.result_types:
            f1score = f1_score(label, pred, average='macro')
            self.result_dict['f1score'] += f1score
        if 'bal_acc' in self.result_types:
            bal_acc = balanced_accuracy_score(label, pred)
            self.result_dict['bal_acc'] += bal_acc
        if 'acc' in self.result_types:
            acc = accuracy_score(label, pred)
            self.result_dict['acc'] += acc

    def one_iter(self, data, class_weight=None, augmentation=False):
        image, image_mask, tab, tab_mask, label = self.prepare_data(data, augmentation=augmentation)

        label = torch.stack(label).permute(1, 0).reshape(-1)
        pred, _, _, _ = self.net(image, image_mask, tab, tab_mask, is_train=augmentation)
        pred = pred.view(-1, pred.size(-1))

        pred = pred[label != -1]        # remove those that do not have a ground truth label
        label = label[label != -1]
        cls_loss = F.cross_entropy(pred, label, weight=class_weight) if len(label) > 0 else 0
        self.loss_dict['cls'] += cls_loss
        return cls_loss, nn.Softmax(dim=1)(pred), label

    def train(self, train_loader, val_loader, optimizer, scheduler, class_weight, params):
        self.set_train()
        for self.epoch in range(self.config['training']['epoch']):
            self.clear_loss()
            self.clear_result()
            all_pred, all_label = [], []
            for batch_idx, data in enumerate(train_loader):
                self.set_zero_grad()
                train_loss, pred, label = self.one_iter(data, class_weight, augmentation=self.config['training']['augmentation'])
                train_loss.backward()
                if self.config['training']['grad_clip'] > 0:
                    clip_grad_norm_(params, self.config['training']['grad_clip'])
                for i in range(len(optimizer)):
                    optimizer[i].step()
                for i in range(len(scheduler)):
                    scheduler[i].step()
                all_pred.extend(pred.data.cpu().numpy().tolist())
                all_label.extend(label.data.cpu().numpy().tolist())
            self.get_result(all_pred, all_label)
            if (self.epoch + 1) % self.config['eval_freq'] == 0:
                prefix = "Epoch : {:.0f}/{:.0f}\n  Train".format((self.epoch + 1), self.config['training']['epoch'])
                self.print_loss_result('train', prefix, len(train_loader), is_train=True)
                self.eval(val_loader, is_train=True)    # evalution

    def eval(self, data_loader, is_train):
        self.set_eval()
        with torch.no_grad():
            self.clear_loss()
            self.clear_result()
            all_pred, all_label = [], []
            for batch_idx, data in enumerate(data_loader):
                self.set_zero_grad()
                _, pred, label = self.one_iter(data, augmentation=False)
                all_pred.extend(pred.data.cpu().numpy().tolist())
                all_label.extend(label.data.cpu().numpy())
            self.get_result(all_pred, all_label)
            self.print_loss_result('val', prefix="  Evaluation", length=len(data_loader), is_train=is_train)

        if is_train:
            t = 'ap' if 'ap' in self.result_types else 'bal_acc'
            if self.result_dict[t] > self.best:
                self.best = self.result_dict[t]
                self.save_networks(save_path=os.path.join(self.ckp_folder, 'best_' + t + '.pth'))
        self.set_train()
        return self.result_dict


