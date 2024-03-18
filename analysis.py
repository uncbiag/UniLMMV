import os
import torch
import warnings
import argparse
import numpy as np
import torch.nn as nn
import seaborn as sns
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 22})

from run import load_data
from trainer import Trainer
from network.model import Model
from config import get_default_config
from utils import setup, set_seed, create_save_folder


def evaluation(trainer):
    trainer.set_eval()
    all_pred, all_label = [], []
    with torch.no_grad():
        trainer.clear_loss()
        for batch_idx, data in enumerate(test_loader):
            trainer.set_zero_grad()
            image, image_mask, tab, tab_mask, label = trainer.prepare_data(data, augmentation=False)
            label = torch.stack(label).permute(1, 0).reshape(-1)
            pred, _, _, _ = trainer.net(image, image_mask, tab, tab_mask, is_train=False)
            pred = pred.view(-1, pred.size(-1))
            pred = nn.Softmax(dim=1)(pred)
            pred = pred[label != -1]  # remove those that do not have a ground truth label
            label = label[label != -1]
            all_pred.extend(pred.data.cpu().numpy().tolist())
            all_label.extend(label.data.cpu().numpy())
    trainer.clear_result()
    trainer.get_result(all_pred, all_label)
    result_dict = {key: round(trainer.result_dict[key], 4) for key in trainer.result_dict}
    return result_dict


def eval_longitudinal(trainer):
    trainer.set_eval()
    result = dict()
    with torch.no_grad():
        for t in range(num_time):
            trainer.clear_loss()
            all_pred, all_label = [], []
            for batch_idx, data in enumerate(test_loader):
                trainer.set_zero_grad()
                image, image_mask, tab, tab_mask, label = trainer.prepare_data(data, augmentation=False)

                for i in range(t, -1, -1):
                    if image is not None:
                        for j in range(len(image)):     # loop modality
                            image[j][t - i] = image[j][num_time - i - 1]        # t0=t5; t0=t4 t1=t5; ...
                            image_mask[j][t - i] = image_mask[j][num_time - i - 1]
                    if tab is not None:
                        tab[t - i] = tab[num_time - i - 1]
                        tab_mask[t - i] = tab_mask[num_time - i - 1]
                    label[t - i] = label[num_time - i - 1]

                pred, _, _, _ = trainer.net(image, image_mask, tab, tab_mask, is_train=False)
                pred = pred.permute(1, 0, 2)[t]
                pred = nn.Softmax(dim=1)(pred)
                label = torch.stack(label)[t]
                pred = pred[label != -1]  # remove those that do not have a ground truth label
                label = label[label != -1]
                all_pred.extend(pred.data.cpu().numpy().tolist())
                all_label.extend(label.data.cpu().numpy().tolist())

            trainer.clear_result()
            trainer.get_result(all_pred, all_label)
            result_dict = {key: round(trainer.result_dict[key], 4) for key in trainer.result_dict}
            result[t] = result_dict
    return result


def drop_modality(views, image, image_mask, tab, tab_mask, drop):
    if 'tab' in drop and 'tab' in views:
        tab = None
        tab_mask = None
        views.remove('tab')
        drop.remove('tab')
    if len(drop) > 0:
        for d in drop:
            idx = views.index(d)
            if 'tab' in views:
                idx = idx - 1 if idx > views.index('tab') else idx
            for i in range(len(image[idx])):
                if image[idx][i] is not None:
                    image[idx][i] = None
                    image_mask[idx][i] = torch.zeros_like(image_mask[idx][i]).to(image_mask[idx][i].device)
    return image, image_mask, tab, tab_mask


def eval_pruning(trainer):
    trainer.set_eval()
    all_label = []
    all_pred = dict()
    for modality in prune_modalities:
        name = '-'.join(str(m) for m in modality)
        all_pred[name] = []
    result = dict()

    with torch.no_grad():
        trainer.clear_loss()
        for batch_idx, data in enumerate(test_loader):
            trainer.set_zero_grad()
            for i in range(len(prune_modalities)):
                name = '-'.join(str(m) for m in prune_modalities[i])
                image, image_mask, tab, tab_mask, label = trainer.prepare_data(data, augmentation=False)
                label = torch.stack(label).permute(1, 0).reshape(-1)
                if i == 0:
                    label_cp = label.clone()
                    all_label.extend(label_cp[label_cp != -1].data.cpu().numpy())
                image, image_mask, tab, tab_mask = drop_modality(config['views'].copy(), image, image_mask, tab, tab_mask, drop=prune_modalities[i].copy())
                pred, _, _, _ = trainer.net(image, image_mask, tab, tab_mask, is_train=False)
                pred = pred.view(-1, pred.size(-1))
                pred = nn.Softmax(dim=1)(pred)
                pred = pred[label != -1]  # remove those that do not have a ground truth label
                all_pred[name].extend(pred.data.cpu().numpy().tolist())

    for p in all_pred.keys():
        trainer.clear_result()
        trainer.get_result(all_pred[p], all_label)
        result_dict = {key: round(trainer.result_dict[key], 4) for key in trainer.result_dict}
        result[p] = result_dict
    return result


def eval_importance(trainer):
    # find the most important modality
    trainer.set_eval()
    all_label, all_name = [], []
    all_pred = dict()
    for p in config['views']:
        if p == 'TC_thickness':     # TC is merged with FC
            continue
        all_pred[p] = []
    num_modality = len(config['views']) - 1 if 'TC_thickness' in config['views'] else len(config['views'])
    # [num_modality, label, names]
    result_name = [[[] for _ in range(config['attention']['output_dim'])] for _ in range(num_modality)]

    with torch.no_grad():
        trainer.clear_loss()
        for batch_idx, data in enumerate(test_loader):
            trainer.set_zero_grad()
            # loop modality
            for i, p in enumerate(config['views']):
                if p == 'TC_thickness':
                    continue
                image, image_mask, tab, tab_mask, label = trainer.prepare_data(data, augmentation=False)
                label = torch.stack(label).permute(1, 0).reshape(-1)
                if i == 0:
                    label_cp = label.clone()
                    full_name = []
                    for n, name in enumerate(data['name']):
                        lst = []
                        for m, month in enumerate(list(data['label'].keys())):
                            lst.append(name + '_' + month)
                        full_name.append(lst)
                    full_name = np.array(full_name).reshape(-1)
                    label_cp = label_cp.data.cpu().numpy()
                    all_name.extend(full_name[label_cp != -1])
                    all_label.extend(label_cp[label_cp != -1])
                drop = [p]
                if p == 'FC_thickness':
                    drop.append('TC_thickness')
                image, image_mask, tab, tab_mask = drop_modality(config['views'].copy(), image, image_mask, tab, tab_mask, drop=drop)
                pred, _, _, _ = trainer.net(image, image_mask, tab, tab_mask, is_train=False)
                pred = pred.view(-1, pred.size(-1))
                pred = nn.Softmax(dim=1)(pred)
                pred = pred[label != -1]  # remove those that do not have a ground truth label
                all_pred[p].extend(pred.data.cpu().numpy().tolist())

    # dict to numpy
    all_pred_np = []
    for p in all_pred.keys():
        all_pred_np.append(all_pred[p])
    all_pred_np = np.array(all_pred_np)
    for i in range(len(all_pred_np[0])):  # loop every instance
        label = all_label[i]
        # pick the one with the worst prediction (which modality is the most importance)
        idx = np.where(all_pred_np[:, i, label] == min(all_pred_np[:, i, label]))[0][0]
        result_name[idx][label].append(all_name[i])
    return result_name


def plot_class_heatmap(result):
    # plot heatmap: for each modality, the percentage of data view it as the most important
    modality = config['views'].copy()
    if 'TC_thickness' in modality:
        modality.remove('TC_thickness')
    modality = list(map(lambda x: x.replace('FC_thickness', 'C'), modality))
    modality = list(map(lambda x: x.replace('knee', 'K'), modality))
    modality = list(map(lambda x: x.replace('pelvis', 'P'), modality))
    modality = list(map(lambda x: x.replace('tab', 'T'), modality))

    result_count = [[[len(result[i][j][k]) for k in range(len(result[i][j]))] for j in
                     range(len(result[i]))] for i in range(len(result))]
    print(np.round(np.mean(result_count, axis=0), 4))
    plt.clf()
    percentage = np.mean(result_count, axis=0)
    percentage = percentage / np.sum(percentage, axis=0)
    # flip knee & cartilage
    if 'K' in modality and 'C' in modality:
        idxk = modality.index('K')
        idxc = modality.index('C')
        temp = percentage[idxk].copy()
        percentage[idxk] = percentage[idxc]
        percentage[idxc] = temp
        modality[idxk] = 'C'
        modality[idxc] = 'K'
    if 'KLG' in config['ex_name']:
        xticklabels = ['01', '2', '3', '4']
        xlabel = 'KLG'
    if 'Pain' in config['ex_name']:
        xticklabels = ['no', 'yes']
        xlabel = 'Pain'
    sns.heatmap(percentage, annot=True, fmt='.2f', xticklabels=xticklabels, yticklabels=modality, cmap='Blues')
    plt.xlabel(xlabel)
    plt.ylabel('Modality')
    plt.savefig('./heatmap_' + config['ex_name'] + '.png')
    return result_count     # beaware the order is TKCP


def main():
    if args.ex_num == 0:
        results0 = dict()
        for rt in result_types:
            results0[rt] = []
    if args.ex_num == 1:
        results1 = dict()
        for rt in result_types:
            results1[rt] = {k: [] for k in range(num_time)}
    if args.ex_num == 2:
        results2 = dict()
        for rt in result_types:
            results2[rt] = dict()
            for modality in prune_modalities:
                name = '-'.join(str(m) for m in modality)
                results2[rt][name] = []
    if args.ex_num == 3:
        results3 = []

    for seed in args.seeds:
        set_seed(seed)
        config['save_folder'] = create_save_folder(config, seed)  # here the folder is for load checkpoint

        # Load checkpoint
        t = 'ap' if 'ap' in result_types else 'bal_acc'
        ckp_path = os.path.join(config['save_folder'], 'checkpoint', 'best_' + t + '.pth')
        net.load_state_dict(torch.load(ckp_path), strict=False)
        print('loaded checkpoint from: ', ckp_path)

        # build trainer
        trainer = Trainer(config, net, logger, result_types=result_types)

        if args.ex_num == 0:     # test the result
            result_dict = evaluation(trainer)
            for k in results0.keys():
                results0[k].append(result_dict[k])

        if args.ex_num == 1:     # varies number of input timpoints
            result_dict = eval_longitudinal(trainer)
            for k1 in result_dict.keys():
                for k2 in results1.keys():
                    results1[k2][k1].append(result_dict[k1][k2])

        if args.ex_num == 2:     # prune modality
            result_dict = eval_pruning(trainer)
            for k1 in result_dict.keys():
                for k2 in results2.keys():
                    results2[k2][k1].append(result_dict[k1][k2])

        if args.ex_num == 3:     # modality importance
            results3.append(eval_importance(trainer))

    # print result
    if args.ex_num == 0:
        for key in results0.keys():
            print(' ', key, ': ', results0[key], '\n   mean: ', np.round(np.mean(results0[key]), 4),
                  ', std: ', np.round(np.std(results0[key]), 4))
    if args.ex_num == 1:
        for i in range(num_time):
            print('Pred 72m with ', i+1, ' timepoints')
            for key in results1.keys():
                print(' ', key, ': ', results1[key][i], '\n   mean: ', np.round(np.mean(results1[key][i]), 4),
                      ', std: ', np.round(np.std(results1[key][i]), 4))
    if args.ex_num == 2:
        for p in results2[result_types[0]].keys():
            print('Missing modality', p, ': ')
            for key in results2.keys():
                print(' ', key, ': ', results2[key][p], '\n   mean: ', np.round(np.mean(results2[key][p]), 4),
                      ', std: ', np.round(np.std(results2[key][p]), 4))
    if args.ex_num == 3:
        result_count = plot_class_heatmap(results3)       # be aware the order is TKCP
        # print total number of data for each modality
        result_count = np.sum(result_count, axis=-1)
        for i in range(len(result_count[0])):
            print('mean: ', np.round(np.mean(result_count[:, i]), 4),
                  'std: ', np.round(np.std(result_count[:, i]), 4))


if __name__ == '__main__':
    warnings.simplefilter("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('--devices', type=str, default='0', help='gpu device ids')
    parser.add_argument('--eval_freq', type=int, default=1, help='gap evaluations')
    parser.add_argument('--seeds', type=int, default=[12, 21, 23, 42, 123], nargs='+', help='seed')
    parser.add_argument('--views', type=str, default=['tab', 'knee', 'FC_thickness', 'TC_thickness', 'pelvis'],
                        nargs='+', help='tab, knee, FC_thickness, TC_thickness, pelvis')
    parser.add_argument('--ex_num', type=int, default=0,
                        help='0: test the result, 1: varies num of timpoints, 2: prune view, 3: modality importance')
    args = parser.parse_args()

    result_types = ['ap', 'roc', 'f1score', 'bal_acc']
    plt.rcParams["figure.figsize"] = [8.8, 6.6]
    # use for experiment 2
    prune_modalities = [['pelvis'], ['FC_thickness', 'TC_thickness'], ['knee'],
                        ['FC_thickness', 'TC_thickness', 'pelvis'], ['knee', 'pelvis'], ['knee', 'FC_thickness', 'TC_thickness'], ['tab', 'pelvis']]      # TKC, KC, TC

    config = get_default_config()
    config, logger, device = setup(args, config)
    num_time = 6 if 'Pain' in config['ex_name'] else 5
    test_loader, tab_info = load_data(config, phase=['test'])
    test_loader = test_loader[0]

    # Build model
    net = Model(config, cat_dim=tab_info[0], num_con=tab_info[1])
    net = nn.DataParallel(net)
    net.to(device)

    main()

