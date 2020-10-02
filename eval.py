import os, shutil
import json
import os.path as osp
import re
import logging
import time
import random
from functools import reduce
import resource
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

import numpy as np
import scipy as sp
from scipy.spatial.distance import pdist, squareform
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision.models import resnet18

from model.misc.utils import init, RunningAverage, ShouldSaveModel, myself, prepare_train, save_checkpoint, \
    longtensor_to_one_hot, EarlyStop, wait_gpu, worker_init_fn_seed, elapsed_timer, load_checkpoint, plot_grad_flow
from model.datasets.glove import load_glove_as_dict
from model.datasets.CompositionDataset import CompositionDataset
from model.SepMask import SepMix, Discriminator
from model.pygcn import normalize

tqdm_commons = {'ncols': 100, 'ascii': True, 'leave': True}

if 'NO_GPU_WAIT' not in os.environ.keys():
    wait_gpu(req_mem=int(os.environ.get('REQ_MEM', '4000')))


def params(p):

    p.add_argument('--dataset', choices=['mitstates', 'ut-zap50k'],
                   default='ut-zap50k', help='Dataset for training and testing.')
    p.add_argument('--data_path', default='.', help='Path where you place your dataset.')
    p.add_argument('--split', choices=['compositional-split', 'natural-split'], default='compositional-split')
    p.add_argument('--batch-size', '--batch_size', type=int, default=512)
    p.add_argument('--test-batch-size', '--test_batch_size', type=int, default=32)
    p.add_argument('--model_path', type=str)
    p.add_argument('--latent_dims', type=int, default=512)
    p.add_argument('--topk', type=int, default=1)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--kneg', type=int, default=1)
    p.add_argument('--num_workers', type=int, default=5)
    p.add_argument('--pre_feat', action='store_true', default=False)
    p.add_argument('--debug_val', '--debug-val',
                   action='store_true', default=False)
    
    return p

def sim(x, y): return -(x-y).norm(p=2, dim=1).unsqueeze(1)

def h_mean(a, b):
    return (2*a*b) / (a+b+1e-8)

def val_separate(model, dataloader, phase='val', topk=1, **kwargs):
    args  = kwargs['args']
    model.eval()
    total_count = 0
    
    correct_unseen = 0
    o_close = 0
    a_close = 0
    correct_seen = 0
    o_open = 0
    a_open = 0

    train_pairs = dataloader.dataset.train_pairs
    test_pairs  = dataloader.dataset.val_pairs if dataloader.dataset.phase == 'val' else dataloader.dataset.test_pairs

    with torch.no_grad():
        test_sample_num = len(dataloader.dataset)

        seen_pairs   = sorted(list(set(train_pairs).intersection(test_pairs)))
        unseen_pairs = sorted(list(set(test_pairs) - set(train_pairs)))
        pair_data_seen_att  = np.zeros([len(seen_pairs), kwargs['complex_dim']])
        pair_data_seen_obj  = np.zeros([len(seen_pairs), kwargs['complex_dim']])
        pair_data_unseen_att = np.zeros([len(unseen_pairs), kwargs['complex_dim']])
        pair_data_unseen_obj = np.zeros([len(unseen_pairs), kwargs['complex_dim']])
        
        test_data_att = np.zeros([test_sample_num, kwargs['complex_dim']])
        test_data_obj = np.zeros([test_sample_num, kwargs['complex_dim']])
        i = 0
        for _, data in tqdm(enumerate(dataloader), desc='GT Feature', total=len(dataloader), disable=args.no_pbar, **tqdm_commons):
            if args.parallel:
                output = model.module.forward(data[0].float(), None, None, status='eval', vis_backbone=kwargs['vis_backbone'])
            else:
                output = model.forward(data[0].float(), None, None, status='eval', vis_backbone=kwargs['vis_backbone'])
            feat_tmp = output['im_feat']
            test_data_att[i:i + feat_tmp.shape[0], :] = output['im_att_feat'].detach().cpu().numpy()
            test_data_obj[i:i + feat_tmp.shape[0], :] = output['im_obj_feat'].detach().cpu().numpy()
            i += dataloader.batch_size
            if args.debug_mode:
                break
        
        for i in range(0, len(unseen_pairs)):
            att_idx = torch.Tensor([dataloader.dataset.attr2idx[unseen_pairs[i][0]]]).long()
            obj_idx = torch.Tensor([dataloader.dataset.obj2idx[unseen_pairs[i][1]]]).long()
            if args.parallel:
                output = model.module.forward(data[0][0].unsqueeze(0).float(), att_idx, obj_idx, mask_target=True, status='eval', ignore_img=True)
            else:
                output = model.forward(data[0][0].unsqueeze(0).float(), att_idx, obj_idx, mask_target=True, status='eval', ignore_img=True)
            pair_data_unseen_att[i, :] = output['lin_att_values'].detach().cpu().numpy()
            pair_data_unseen_obj[i, :] = output['lin_obj_values'].detach().cpu().numpy()
            if args.debug_mode:
                break
        
        for i in range(0, len(seen_pairs)):
            att_idx = torch.Tensor([dataloader.dataset.attr2idx[seen_pairs[i][0]]]).long()
            obj_idx = torch.Tensor([dataloader.dataset.obj2idx[seen_pairs[i][1]]]).long()
            if args.parallel:
                output = model.module.forward(data[0][0].unsqueeze(0).float(), att_idx, obj_idx, status='eval', mask_target=True, ignore_img=True)
            else:
                output = model.forward(data[0][0].unsqueeze(0).float(), att_idx, obj_idx, status='eval', mask_target=True, ignore_img=True)
            pair_data_seen_att[i, :] = output['lin_att_values'].detach().cpu().numpy()
            pair_data_seen_obj[i, :] = output['lin_obj_values'].detach().cpu().numpy()
            if args.debug_mode:
                break

        pair_t_unseen_att = torch.FloatTensor(pair_data_unseen_att).cuda()
        pair_t_seen_att = torch.FloatTensor(pair_data_seen_att).cuda()
        pair_t_att = torch.cat((pair_t_unseen_att, pair_t_seen_att))
        pair_t_unseen_obj = torch.FloatTensor(pair_data_unseen_obj).cuda()
        pair_t_seen_obj = torch.FloatTensor(pair_data_seen_obj).cuda()
        pair_t_obj = torch.cat((pair_t_unseen_obj, pair_t_seen_obj))

        dist = torch.zeros(test_sample_num, len(unseen_pairs) + len(seen_pairs))
        STEPS = 50
        correct_unseen = torch.zeros(STEPS, )
        total_unseen = 0
        correct_seen = torch.zeros(STEPS, )
        total_seen = 0

        for i in tqdm(range(0, test_sample_num), disable=args.no_pbar, **tqdm_commons):
            dist[i] = sim(pair_t_att, torch.Tensor(test_data_att[i, :]).cuda().repeat(pair_t_att.shape[0], 1)).squeeze() + \
                    sim(pair_t_obj, torch.Tensor(test_data_obj[i, :]).cuda().repeat(pair_t_obj.shape[0], 1)).squeeze()
        
        dist_diff = dist.max() - dist.min()
        biases = torch.linspace(-dist_diff-0.1, dist_diff+0.1, STEPS)
        for i in tqdm(range(0, test_sample_num), disable=args.no_pbar, **tqdm_commons):
            _, att_gt, obj_gt = dataloader.dataset.data[i]

            is_seen = (att_gt, obj_gt) in seen_pairs
            if is_seen:
                total_seen += 1
            else:
                total_unseen += 1
        
            for ii, bias in enumerate(biases):
                dist_bias = dist[i].clone()
                dist_bias[:len(unseen_pairs)] += bias
                preds = dist_bias.argsort(dim=0)[-topk:]
                
                for pred in preds:
                    pred_pairs = (unseen_pairs + seen_pairs)[pred]
                    correct = int(pred_pairs[0] == att_gt and pred_pairs[1] == obj_gt)
                    if is_seen:
                        correct_seen[ii] += correct
                    else:
                        correct_unseen[ii] += correct
                    if correct == 1:
                        continue

            if args.debug_mode:
                break
        correct_unseen /= total_unseen
        correct_seen  /= total_seen
        auc = torch.trapz(correct_seen, correct_unseen)
   
    test_info = {
        'phase':    phase,
        'auc':      float(auc),
        'seen_acc':   float(correct_seen.max()),
        'unseen_acc': float(correct_unseen.max()),
        'h_mean':   float(h_mean(correct_unseen, correct_seen).max())
    }
    
    return test_info

if __name__ == '__main__':
    args = init(user_param=params)

    obj_class_num    = {'ut-zap50k': 12,  'mitstates': 245}
    att_class_num    = {'ut-zap50k': 16,  'mitstates': 115}
    obj_encode_dims  = {'ut-zap50k': 300, 'mitstates': 300}
    att_encode_dims  = {'ut-zap50k': 300, 'mitstates': 300}

    glove_embedding  = load_glove_as_dict(f'{args.data_path}/glove', dimension=300, identifier='42B')

    train_dataset    = CompositionDataset(f'{args.data_path}/{args.dataset}', 'train', split=args.split, embedding_dict=glove_embedding, kneg=args.kneg, precompute_feat=args.pre_feat)
    test_dataset     = CompositionDataset(f'{args.data_path}/{args.dataset}', 'test',  split=args.split, embedding_dict=glove_embedding, precompute_feat=args.pre_feat)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn_seed(args), drop_last=True)
    test_dataloader  = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn_seed(args), drop_last=False)
    if args.split == 'compositional-split':
        val_dataset    = test_dataset
        val_dataloader = test_dataloader
    elif args.split == 'natural-split':
        val_dataset    = CompositionDataset(f'{args.data_path}/{args.dataset}', 'val',  split=args.split, embedding_dict=glove_embedding, precompute_feat=args.pre_feat)
        val_dataloader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn_seed(args), drop_last=False)

    def _emb(s):
        ss = re.split('\.|-', s)
        emb = np.zeros(glove_embedding['the'].shape)
        if len(ss) == 1:
            try:
                emb = glove_embedding[ss[0]]
            except KeyError:
                logging.warning(
                    f'Cannot embed word \"{ss[0]}\", fallback to <unk>')
                emb = glove_embedding['<unk>']
        else:
            for w in ss:
                try:
                    emb += glove_embedding[w]
                except KeyError:
                    logging.warning(
                        f'Cannot embed word \"{w}\", fallback to <unk>')
                    emb += glove_embedding['<unk>']
        return emb

    att_emb_dict = {k: v for (k, v) in [(
        kk, _emb(kk.lower())) for kk in train_dataset.attrs]}
    obj_emb_dict = {k: v for (k, v) in [(
        kk, _emb(kk.lower())) for kk in train_dataset.objs]}
    train_dataset.att_emb_dict = att_emb_dict
    train_dataset.obj_emb_dict = obj_emb_dict
    test_dataset.att_emb_dict  = att_emb_dict
    test_dataset.obj_emb_dict  = obj_emb_dict
    val_dataset.att_emb_dict  = att_emb_dict
    val_dataset.obj_emb_dict  = obj_emb_dict
    
    att_emb = np.array([v for (_, v) in att_emb_dict.items()])
    att_adj = squareform(1-pdist(att_emb, 'cosine'))

    obj_emb = np.array([v for (_, v) in obj_emb_dict.items()])
    obj_adj = squareform(1-pdist(obj_emb, 'cosine'))

    seen_mask = torch.zeros((att_class_num[args.dataset]+obj_class_num[args.dataset], att_class_num[args.dataset]+obj_class_num[args.dataset]))
    for seen_pair in train_dataset.train_pairs:
        att_idx, obj_idx = train_dataset.attr2idx[seen_pair[0]], train_dataset.obj2idx[seen_pair[1]]
        seen_mask[att_idx, att_class_num[args.dataset]+obj_idx] = 1
        seen_mask[att_class_num[args.dataset]+obj_idx, att_idx] = 1
    seen_mask[:att_class_num[args.dataset], :att_class_num[args.dataset]] = 1
    seen_mask[att_class_num[args.dataset]:, att_class_num[args.dataset]:] = 1

    model_config = {
        'complex_dim':     args.latent_dims,
        'primitive_dim':   512,
        'seen_mask':       seen_mask == 1,
        'obj_encodes':     torch.Tensor(obj_emb).cuda(),
        'att_encodes':     torch.Tensor(att_emb).cuda(),
        'obj_encode_dim':  obj_encode_dims[args.dataset],
        'att_encode_dim':  att_encode_dims[args.dataset],
        'obj_class_num':   obj_class_num[args.dataset],
        'att_class_num':   att_class_num[args.dataset],
        'obj_adj':         torch.Tensor(normalize(obj_adj)).cuda(),
        'att_adj':         torch.Tensor(normalize(att_adj)).cuda(),
        'dropout':         args.dropout,
        'args':            args
    }
    
    device = torch.device('cuda')

    model = SepMix(**model_config).cuda()
    pretrained = torch.load(args.model_path)
    model.load_state_dict(pretrained['model'], strict=False)

    vis_backbone = None 

    val = val_separate  # val_separate val_distance
    if args.test_only:
        test_info = val(model, test_dataloader, topk=args.topk, phase='test', device=device, complex_dim=args.latent_dims, vis_backbone=vis_backbone, args=args)
        print(test_info)
        print()
        test_info = val(model, val_dataloader, topk=args.topk, phase='val', device=device, complex_dim=args.latent_dims, vis_backbone=vis_backbone, args=args)
        print(test_info)