import os
import gc
import torch
import argparse
import warnings
import numpy as np
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.utils.class_weight import compute_class_weight


from network.model import Model
from trainer import Trainer
from dataset import get_loader
from config import get_default_config
from utils import setup, set_seed, create_save_folder, get_WOMAC_label, get_KLG_label


def get_class_weight(sheet_name, train_loader):
    all_labels = []
    for batch_idx, data in enumerate(train_loader):
        if 'WOMAC' in sheet_name:
            label = get_WOMAC_label(data['label'])
        else:
            label = get_KLG_label(data['label'])
        for month in list(label.keys()):
            all_labels.extend(label[month].cpu().numpy())
    all_labels = [l for l in all_labels if l != -1]     # remove label do not exist
    class_weight = compute_class_weight(class_weight='balanced', classes=np.unique(all_labels), y=np.array(all_labels))
    class_weight = torch.tensor(class_weight, dtype=torch.float).cuda()
    return class_weight


def load_data(config, phase):
    data_loader = []
    for p in phase:
        shuffle = True if p == 'train' else False
        loader, tab_info = get_loader(config, phase=p, shuffle=shuffle)
        data_loader.append(loader)
    return data_loader, tab_info


def set_optim_sched(config, net, steps_per_epoch):
    views = config['views'].copy()

    params_cls = list(net.module.classifier.parameters()) + list(net.module.atten.parameters()) + list(net.module.empty_emb.parameters())
    optimizer_cls = AdamW(params_cls, lr=config['training']['lr_cls'], weight_decay=0.001)
    scheduler_cls = OneCycleLR(optimizer_cls, max_lr=config['training']['lr_cls'],
                               steps_per_epoch=steps_per_epoch, epochs=config['training']['epoch'])

    param = params_cls
    optimizer = [optimizer_cls]
    scheduler = [scheduler_cls]

    if 'tab' in views:
        params_tab = list(net.module.encoder_tab.parameters())
        optimizer_tab = AdamW(params_tab, lr=config['training']['lr_tab'], weight_decay=0.001)
        scheduler_tab = OneCycleLR(optimizer_tab, max_lr=config['training']['lr_tab'],
                                   steps_per_epoch=steps_per_epoch, epochs=config['training']['epoch'])
        param += params_tab
        optimizer.append(optimizer_tab)
        scheduler.append(scheduler_tab)
        views.remove('tab')

    params_xray, params_thickness = [], []
    if len(views) > 0:
        for i in range(len(views)):
            if 'thickness' in views[i]:
                params_thickness += list(getattr(net.module, 'encoder_img'+str(i+1)).parameters())
            else:
                params_xray += list(getattr(net.module, 'encoder_img'+str(i+1)).parameters())

    if len(params_xray) > 0:
        optimizer_xray = AdamW(params_xray, lr=config['training']['lr_xray'], weight_decay=0.001)
        scheduler_xray = OneCycleLR(optimizer_xray, max_lr=config['training']['lr_xray'],
                                    steps_per_epoch=steps_per_epoch, epochs=config['training']['epoch'])
        param += params_xray
        optimizer.append(optimizer_xray)
        scheduler.append(scheduler_xray)

    if len(params_thickness) > 0:
        optimizer_thickness = AdamW(params_thickness, lr=config['training']['lr_thickness'], weight_decay=0.001)
        scheduler_thickness = OneCycleLR(optimizer_thickness, max_lr=config['training']['lr_thickness'],
                                         steps_per_epoch=steps_per_epoch, epochs=config['training']['epoch'])
        param += params_thickness
        optimizer.append(optimizer_thickness)
        scheduler.append(scheduler_thickness)

    return optimizer, scheduler, param


def main():
    config = get_default_config()
    config, logger, device = setup(args, config)

    for seed in args.seeds:
        set_seed(seed)
        config['save_folder'] = create_save_folder(config, seed)

        # data loader
        loader, tab_info = load_data(config, phase=['train', 'val'])
        train_loader, val_loader = loader

        # Build model
        net = Model(config, cat_dim=tab_info[0], num_con=tab_info[1])
        net = nn.DataParallel(net)
        net.to(device)

        # Set optimizer and scheduler
        optimizer, scheduler, param = set_optim_sched(config, net, steps_per_epoch=len(train_loader))

        # get class weight
        class_weight = get_class_weight(config['data']['label_sheet_name'], train_loader)
        logger.info(f'class weights: {class_weight}')

        # build trainer
        trainer = Trainer(config, net, logger, result_types=result_types)

        # train
        trainer.train(train_loader, val_loader, optimizer, scheduler, class_weight, param)
        logger.info('--------------------Training over--------------------')

        # test
        t = 'ap' if 'ap' in result_types else 'bal_acc'
        ckp_path = os.path.join(config['save_folder'], 'checkpoint', 'best_' + t + '.pth')
        net.load_state_dict(torch.load(ckp_path))
        logger.info(f'loaded checkpoint from: {ckp_path}')

        test_loader, _ = load_data(config, phase=['test'])
        result_dict = trainer.eval(test_loader[0], is_train=False)
        logger.info('--------------------Testing over--------------------')
        for key in result_dict.keys():
            results[key].append(result_dict[key])

        gc.collect()
        torch.cuda.empty_cache()

    for key in results:
        logger.info(f'{key}: {results[key]}\n  mean: {np.round(np.mean(results[key]), 4)}, std: {np.round(np.std(results[key]), 4)}')


if __name__ == '__main__':
    warnings.simplefilter("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='gpu device ids')
    parser.add_argument('--eval_freq', type=int, default=1, help='gap evaluations')
    parser.add_argument('--seeds', type=int, default=[12, 21, 23, 42, 123], nargs='+', help='seed')
    parser.add_argument('--views', type=str, default=['tab', 'knee', 'FC_thickness', 'TC_thickness', 'pelvis'],
                        nargs='+', help='tab, knee, FC_thickness, TC_thickness, pelvis')
    args = parser.parse_args()

    result_types = ['ap', 'roc', 'f1score', 'bal_acc', 'acc']
    results = {}
    for rt in result_types:
        results[rt] = []

    main()
