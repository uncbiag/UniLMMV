from __future__ import print_function, absolute_import, division
import os
import sys
import math
import torch
import numpy as np
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import torchvision.models as models

sys.path.append("./network")
from saint import RowColTransformer, embed_data_mask, simple_MLP
from attention import LayerNorm, DecoderBlock, MaskedAttenBlock


class CNNEncoder(nn.Module):
    def __init__(self, arch):
        super(CNNEncoder, self).__init__()
        self.encoder = models.__dict__[arch](pretrained=True)
        self.encoder.fc = nn.Sequential()
        finetune_layers = ['layer2', 'layer3', 'layer4']
        for name, p in self.encoder.named_parameters():
            p.requires_grad = False
            for l in finetune_layers:
                if l in name:
                    p.requires_grad = True
                    break

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        return x


class TabEncoder(nn.Module):
    def __init__(self, config, categories, num_continuous, output_dim):
        super(TabEncoder, self).__init__()
        num_categories = len(categories)
        total_tokens = sum(categories)

        self.simple_MLP = nn.ModuleList([simple_MLP([1, 100, config['dim']]) for _ in range(num_continuous)])
        nfeats = num_categories + num_continuous

        self.mask_embeds_cat = nn.Embedding(num_categories * 2, config['dim'])
        self.mask_embeds_cont = nn.Embedding(num_continuous * 2, config['dim'])

        self.embeds = nn.Embedding(total_tokens, config['dim'])

        categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=0)
        categories_offset = categories_offset.cumsum(dim=-1)[:-1]

        cat_mask_offset = F.pad(torch.Tensor(num_categories).fill_(2).type(torch.int8), (1, 0), value=0)
        cat_mask_offset = cat_mask_offset.cumsum(dim=-1)[:-1]

        con_mask_offset = F.pad(torch.Tensor(num_continuous).fill_(2).type(torch.int8), (1, 0), value=0)
        con_mask_offset = con_mask_offset.cumsum(dim=-1)[:-1]

        self.setup = dict(
            dim=config['dim'],
            categories_offset=categories_offset,
            num_continuous=num_continuous,
            cat_mask_offset=cat_mask_offset,
            con_mask_offset=con_mask_offset,
        )

        self.encoder = RowColTransformer(num_tokens=total_tokens, dim=config['dim'], nfeats=nfeats,
                                         depth=config['n_layers'], heads=config['n_heads'], dim_head=config['head_dim'],
                                         attn_dropout=config['dropout'], ff_dropout=config['dropout'])

        self.fc = nn.Linear(config['dim'], output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

    def forward(self, x_categ, x_cont, cat_mask, con_mask):
        _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,
                                                     self.embeds, self.simple_MLP, self.mask_embeds_cat, self.mask_embeds_cont, self.setup)
        x = self.encoder(x_categ_enc, x_cont_enc)[:, 0, :]
        x = self.relu(self.bn(self.fc(x)))
        return x


class MaskedAtten(nn.Module):
    def __init__(self, config, num_emb):
        super(MaskedAtten, self).__init__()
        self.input_drop = config['input_drop']

        self.sum_emb = nn.Embedding(1, config['input_dim'])
        self.modality_enc = nn.Embedding(num_emb, config['input_dim'])
        self.atten = nn.ModuleList([MaskedAttenBlock(config) for _ in range(config['n_layers'])])
        self.norm = LayerNorm(config['input_dim'], bias=config['bias'])

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config['n_layers']))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, mask, is_train):
        all_att = []
        summarize = self.sum_emb(torch.tensor([0]).cuda())[None, :, :].repeat(x.shape[0], 1, 1)
        mask = torch.cat([torch.tensor([1]).to(torch.int64).cuda()[None, :].repeat(mask.shape[0], 1), mask], dim=1)
        x = torch.cat([summarize, x], dim=1)
        pos = torch.arange(0, x.shape[1], dtype=torch.long, device=x.device)
        modality_emb = self.modality_enc(pos)
        x = x + modality_emb
        if is_train:
            for i in range(len(x[0])-1):        # do not mask out the summarize embedding
                if np.random.choice([0, 1], p=[1-self.input_drop, self.input_drop]):
                    x[:, i+1] = torch.zeros_like(x[:, i+1]).cuda()      # mask the modality
                    mask[:, i+1] = 0        # edit the mask as well
        for block in self.atten:
            x, att = block(x, mask)
            all_att.append(att)
        x = self.norm(x)
        return x[:, 0], all_att


class TransDecoder(nn.Module):
    def __init__(self, config):
        super(TransDecoder, self).__init__()
        self.pos_enc = nn.Embedding(config['block_size'], config['input_dim'])
        self.atten = nn.ModuleList([DecoderBlock(config) for _ in range(config['n_layers'])])
        self.norm = LayerNorm(config['input_dim'], bias=config['bias'])

        self.fc = nn.Linear(config['input_dim'], config['output_dim'], bias=config['bias'])

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config['n_layers']))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):       # x.shape: (len, bs, emb_dim)
        all_att = []
        pos = torch.arange(0, x.shape[1], dtype=torch.long, device=x.device)
        pos_emb = self.pos_enc(pos)  # position embeddings of shape (t, n_embd)
        x = x + pos_emb
        for block in self.atten:
            x, att = block(x)
            all_att.append(att)
        feat = self.norm(x)
        out = self.fc(x)
        return out, feat, all_att


class Model(nn.Module):
    def __init__(self, config, cat_dim=None, num_con=None):
        super(Model, self).__init__()
        views = config['views'].copy()
        self.empty_emb = nn.Embedding(1, config['attention']['input_dim'])

        if 'tab' in views:
            self.encoder_tab = TabEncoder(config['tab_encoder'], categories=tuple(cat_dim), num_continuous=num_con, output_dim=config['attention']['input_dim'])
            views.remove('tab')

        if len(views) > 0:      # images exist
            for i in range(len(views)):
                setattr(self, f'encoder_img{str(i + 1)}', CNNEncoder(config['img_encoder']['arch']))
        self.atten = MaskedAtten(config['attention'], num_emb=len(config['views']) + 1)
        self.classifier = TransDecoder(config['attention'])

    def forward(self, img, img_mask, tab, tab_mask, is_train):
        num_month = len(tab) if tab is not None else len(img[0])
        if tab is not None:
            bs = len(tab[0][0])
            device = tab[0][0].device
        else:
            for i in range(len(img)):
                if img[i][0] is not None:
                    bs = len(img[i][0])
                    device = img[i][0].device
                    break
        features, feature_masks = [], []

        if tab is not None:
            tab = list(zip(*tab))       # [time, 2, bs, dim] -> [2, time, bs, dim]
            tab_mask = list(zip(*tab_mask))     # [time, 2, bs, dim] -> [2, time, bs, dim]
            tab = [torch.cat(tab[i], dim=0) for i in range(len(tab))]       # [2, time, bs, dim] -> [2, time*bs, dim]
            tab_mask = [torch.cat(tab_mask[i], dim=0) for i in range(len(tab_mask))]        # [2, time, bs, dim] -> [2, time*bs, dim]
            feat = self.encoder_tab(tab[0], tab[1], tab_mask[0], tab_mask[1])
            feat = rearrange(feat, '(t b) d -> t b d', t=num_month)
            features.append(feat)
            feature_masks.append(torch.ones((feat.shape[0], feat.shape[1])).to(feat.device))

        if img is not None:
            empty = self.empty_emb(torch.zeros(bs, dtype=torch.long, device=device))      # [bs, dim]

            for i in range(len(img)):
                if None in img[i]:
                    avaliable_idx = [j for j in range(len(img[i])) if img[i][j] is not None]
                    img_i = [img[i][j] for j in range(len(img[i])) if j in avaliable_idx]
                    img_i = torch.cat(img_i, dim=0) if len(avaliable_idx) > 0 else None
                else:
                    avaliable_idx = list(range(len(img[i])))
                    img_i = torch.cat(img[i], dim=0)        # [time, bs, ch, dim, dim] -> [time*bs, ch, dim, dim]
                if img_i is not None:
                    feat = getattr(self, f'encoder_img{str(i+1)}')(img_i)
                    feat = torch.stack(torch.chunk(feat, len(avaliable_idx)), dim=0)        # [time*bs, dim] -> [time, bs, dim]
                count = 0
                new_feat = []
                for j in range(num_month):
                    if j in avaliable_idx:
                        new_feat.append(feat[count])
                        count += 1
                    else:
                        new_feat.append(empty)
                new_feat = torch.stack(new_feat, dim=0)
                features.append(new_feat)
                feature_masks.append(torch.stack(img_mask[i], dim=0))

        features = torch.stack(features, dim=2)      # [modality, time, bs, dim] -> [time, bs, modality, dim]
        features = rearrange(features, 't b m d -> (t b) m d')      # [time, bs, modality, dim] -> [time*bs, modality, dim]
        feature_masks = torch.stack(feature_masks, dim=-1)   # [modality, time, bs] -> [time, bs, modality]
        feature_masks = rearrange(feature_masks, 't b m -> (t b) m')        # [time, bs, modality] -> [time*bs, modality]
        latent, atten = self.atten(features, feature_masks, is_train)
        latent = rearrange(latent, '(t b) d -> t b d', t=num_month)      # [time*bs, dim] -> [time, bs, dim]
        atten_views = rearrange(atten, 'l (t b) h m1 m2 -> l t b h m1 m2', t=num_month)      # [layers, time*bs, heads, modalities+1, modalities+1] -> [layers, time, bs, heads, modalities+1, modalities+1]
        pred, feat, atten_month = self.classifier(latent.permute(1, 0, 2))
        return pred, feat, atten_views, atten_month



