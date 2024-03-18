import os
import torch
import logging
import random
import datetime
import numpy as np
from monai.transforms import *
from skimage.transform import resize


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_logger():
    os.makedirs('./logs', exist_ok=True)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(
        './logs/' + str(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H-%M-%S')) + '.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def setup(args, config):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    config['eval_freq'] = args.eval_freq
    config['views'] = args.views
    logger = get_logger()
    for (k, v) in config.items():
        if isinstance(v, dict):
            logger.info("%s={" % (k))
            for (g, z) in v.items():
                logger.info("          %s = %s" % (g, z))
        else:
            logger.info("%s = %s" % (k, v))
    return config, logger, device


def create_save_folder(config, seed):
    views = '-'.join(str(x) for x in config['views'])
    save_folder = '{}/{}/{}/seed{}/{}_img{}_e{}_bs{}_lr{}_{}_{}_{}_aug{}_drop{}_load{}'.format(
        config['result_folder'], config['ex_name'], views, seed,
        config['data']['label_sheet_name'].split('.')[0], config['data']['img_size'], config['training']['epoch'], config['training']['batch_size'],
        config['training']['lr_cls'], config['training']['lr_tab'], config['training']['lr_xray'], config['training']['lr_thickness'],
        config['training']['augmentation'], config['attention']['input_drop'], config['data']['num_load'])
    os.makedirs(save_folder, exist_ok=True)
    return save_folder


def get_WOMAC_label(label):
    for month in list(label.keys()):
        label[month] = torch.where((label[month] < 5) & (label[month] >= 0), 0, label[month])
        label[month] = torch.where(label[month] >= 5, 1, label[month])
    return label


def get_KLG_label(label):
    for month in list(label.keys()):
        label[month] = torch.where(label[month] <= 0, label[month], label[month] - 1)
    return label


def normalize(img, percentage_clip=-1, max_value=-1, zero_centered=False):
    # normalize into [0, 1] if not zero_centered else [-1, 1]
    img = img - img.min()
    if percentage_clip > 0:
        norm_img = img / np.percentile(img, percentage_clip) * (percentage_clip/100)
    if max_value > 0:
        norm_img = img / (max_value - img.min())
    if zero_centered:
        norm_img = norm_img * 2 - 1
    return norm_img


def resize_image(img, target_size, order=1):
    img = resize(img, target_size, order=order)
    return img


def flip_image(img):
    img = np.fliplr(img)
    return img


def aug_img(img, zero_img, contract_adjust):
    zero_idx = (img.view(img.shape[0], -1).sum(dim=-1) == 0)
    img_sz = img.shape
    rotate = np.pi / 12  # 15 degree
    rotate_range = (rotate, rotate) if len(img_sz) == 3 else (rotate, rotate, rotate)
    if contract_adjust:
        transforms = Compose(
            [RandAffine(prob=0.5, rotate_range=rotate_range, padding_mode='border'),
             RandGaussianNoise(prob=0.5),
             RandAdjustContrast(prob=0.5)])
    else:
        transforms = Compose(
            [RandAffine(prob=0.5, rotate_range=rotate_range, padding_mode='border'),
             RandGaussianNoise(prob=0.5)])
    img = transforms(img)
    img[zero_idx is True] = zero_img
    return img


