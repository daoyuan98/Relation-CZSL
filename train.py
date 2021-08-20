import os, shutil
import json
import os.path as osp
import re
import logging
import time
import random
from functools import reduce
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

import numpy as np
import scipy as sp
from scipy.spatial.distance import pdist, squareform
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch
from torch import nn
# torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision.models import resnet18

from model.misc.utils import init, RunningAverage, ShouldSaveModel, myself, prepare_train, save_checkpoint, \
    longtensor_to_one_hot, EarlyStop, wait_gpu, worker_init_fn_seed, elapsed_timer, load_checkpoint, plot_grad_flow
from model.datasets.glove import load_glove_as_dict
from model.datasets.CompositionDataset import CompositionDataset
from model.SepMask import SepMix
from model.pygcn import normalize

tqdm_commons = {'ncols': 100, 'ascii': True, 'leave': True}

if 'NO_GPU_WAIT' not in os.environ.keys():
    wait_gpu(req_mem=int(os.environ.get('REQ_MEM', '4000')))


def params(p):

    p.add_argument('--dataset', choices=['mitstates', 'ut-zap50k'],
                   default='ut-zap50k', help='Dataset for training and testing.')
    p.add_argument('--data_path', default='.', help='Path where you place your dataset.')
    p.add_argument('--split', choices=['compositional-split', 'natural-split'], default='compositional-split')

    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lr_decay', '--lr-decay', type=float, default=0.1)
    p.add_argument('--batch-size', '--batch_size', type=int, default=512)
    p.add_argument('--test-batch-size', '--test_batch_size', type=int, default=32)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--sch_milestones', type=int, nargs='+', default=[500])
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--loss_weights', type=str, default='{}')

    p.add_argument('--rank_margin', type=float, default=1.0)
    p.add_argument('--latent_dims', type=int, default=512)
    p.add_argument('--kneg', type=int, default=5)

    p.add_argument('--num_workers', type=int, default=5)

    p.add_argument('--meta_samples', type=float, default=0.9)
    p.add_argument('--meta_inclusive', action='store_true', default=True)
    p.add_argument('--pre_feat', action='store_true', default=False)
    p.add_argument('--debug_val', '--debug-val',
                   action='store_true', default=False)
    p.add_argument('--model_dir', type=str, default=".")
    return p


def log_t_loss(neg, pos, anchor, sim=None, margin=None):
    return torch.log(1+torch.exp(margin+sim(neg, anchor)-sim(pos, anchor))).sum() / pos.shape[0]


def t_loss(neg, pos, anchor, sim=None, margin=None):
    return F.relu(margin+sim(neg, anchor)-sim(pos, anchor)).sum() / pos.shape[0]


def log_m_loss(x, anchor, sim=None, margin=None):
    return torch.log(1+torch.exp(margin+sim(x, anchor))).sum() / x.shape[0]


def m_loss(x, anchor, sim=None, margin=None):
    return F.relu(margin+sim(x, anchor)).sum() / x.shape[0]


def sep_loss(a, b):
    return (a * b).norm().sum() / a.shape[0]


def h_mean(a, b):
    return (2*a*b) / (a+b+1e-8)


def sim(x, y): return -(x-y).norm(p=2, dim=1).unsqueeze(1)


def loss_meta(*args, **kwargs):
    output_pp, negs = args
    loss_weights = kwargs['loss_weights']
    rank_margin = kwargs['rank_margin']
    rand_a = kwargs['rand_a']
    rand_o = kwargs['rand_o']
    should_print = kwargs['should_print']

    lp = dict.fromkeys(['ra', 'ro', 'ica', 'ico', 'lca', 'lco'], 0.)
    lp['ra'] = F.mse_loss(output_pp['lin_att_values'], output_pp['masked']['lin_att_values'])
    lp['ro'] = F.mse_loss(output_pp['lin_obj_values'], output_pp['masked']['lin_obj_values'])
    lp['lca'] = F.nll_loss(F.log_softmax(output_pp['masked']['lin_att_logits'], -1), output_pp['masked']['att_idx'])
    lp['lco'] = F.nll_loss(F.log_softmax(output_pp['masked']['lin_obj_logits'], -1), output_pp['masked']['obj_idx'])
    lp['ica'] = F.nll_loss(F.log_softmax(output_pp['masked']['im_att_logits'], -1), output_pp['masked']['att_idx'])
    lp['ico'] = F.nll_loss(F.log_softmax(output_pp['masked']['im_obj_logits'], -1), output_pp['masked']['obj_idx'])

    ln = dict.fromkeys(['ta', 'to', 'ita', 'ito'], 0.)
    for k in range(len(negs)):
        output_pn, output_np = negs[k]
        if rand_a > loss_weights['step_a']:
            ln['ta'] += log_t_loss(output_np['masked']['lin_att_values'], output_pp['masked']['lin_att_values'], output_pp['masked']['im_att_feat'],
                sim=sim, margin=rank_margin)
            ln['ita'] += log_t_loss(output_np['masked']['im_att_feat'], output_pp['masked']['im_att_feat'], output_pp['masked']['lin_att_values'],
            sim=sim, margin=rank_margin)
        if rand_o >loss_weights['step_o']:
            ln['to'] += log_t_loss(output_pn['masked']['lin_obj_values'], output_pp['masked']['lin_obj_values'], output_pp['masked']['im_obj_feat'],
                sim=sim, margin=rank_margin)
            ln['ito'] += log_t_loss(output_pn['masked']['im_obj_feat'], output_pp['masked']['im_obj_feat'], output_pp['masked']['lin_obj_values'],
                sim=sim, margin=rank_margin)
        
    for k in ln.keys():
        ln[k] /= len(negs)
    
    losses = {**lp, **ln}
    pop_keys = []
    for k in losses.keys():
        lw = loss_weights.get(k, 0.0)
        if lw == 0.0 or type(losses[k]) is float:
            pop_keys.append(k)
            continue
        losses[k] *= lw
    for ki in pop_keys:
        losses.pop(ki)

    return losses


def loss_separate(*args, **kwargs):
    output_pp, negs = args
    loss_weights = kwargs['loss_weights']
    rank_margin = kwargs['rank_margin']
    should_print = kwargs['should_print']
    rand_a = kwargs['rand_a']
    rand_o = kwargs['rand_o']

    lp = dict.fromkeys(['ra', 'ro', 'ica', 'ico', 'lca', 'lco'], 0.)

    lp['lca'] = F.nll_loss(F.log_softmax(output_pp['lin_att_logits'], -1), output_pp['att_idx'])
    lp['lco'] = F.nll_loss(F.log_softmax(output_pp['lin_obj_logits'], -1), output_pp['obj_idx'])
    lp['ica'] = F.nll_loss(F.log_softmax(output_pp['im_att_logits'], -1), output_pp['att_idx'])
    lp['ico'] = F.nll_loss(F.log_softmax(output_pp['im_obj_logits'], -1), output_pp['obj_idx'])

    ln = dict.fromkeys(['ta', 'to', 'ita', 'ito'], 0.)
    for k in range(len(negs)):
        output_pn, output_np = negs[k]
        if rand_a > loss_weights['step_a']:
            ln['ta'] += log_t_loss(output_np['lin_att_values'], output_pp['lin_att_values'], output_pp['im_att_feat'],
                sim=sim, margin=rank_margin)
            ln['ita'] += log_t_loss(output_np['im_att_feat'], output_pp['im_att_feat'], output_pp['lin_att_values'],
            sim=sim, margin=rank_margin)
        if rand_o >loss_weights['step_o']:
            ln['to'] += log_t_loss(output_pn['lin_obj_values'], output_pp['lin_obj_values'], output_pp['im_obj_feat'],
                sim=sim, margin=rank_margin)
            ln['ito'] += log_t_loss(output_pn['im_obj_feat'], output_pp['im_obj_feat'], output_pp['lin_obj_values'],
                sim=sim, margin=rank_margin)
        
    for k in ln.keys():
        ln[k] /= len(negs)
    
    losses = {**lp, **ln}
    pop_keys = []
    for k in losses.keys():
        lw = loss_weights.get(k, 0.0)
        if lw == 0.0 or type(losses[k]) is float:
            pop_keys.append(k)
            continue
        losses[k] *= lw
    for ki in pop_keys:
        losses.pop(ki)

    return losses


def val_separate(model, dataloader, phase='val', topk=1, **kwargs):
    args  = kwargs['args']
    model.eval()
    
    correct_unseen = 0
    correct_seen = 0

    train_pairs = dataloader.dataset.train_pairs
    test_pairs  = dataloader.dataset.val_pairs if dataloader.dataset.phase == 'val' else dataloader.dataset.test_pairs

    with torch.no_grad():
        test_sample_num = len(dataloader.dataset)

        seen_pairs   = sorted(list(set(train_pairs).intersection(test_pairs)))
        unseen_pairs = sorted(list(set(test_pairs) - set(train_pairs)))
        # pair_data_seen      = np.zeros([len(seen_pairs), kwargs['complex_dim']])
        pair_data_seen_att  = np.zeros([len(seen_pairs), kwargs['complex_dim']])
        pair_data_seen_obj  = np.zeros([len(seen_pairs), kwargs['complex_dim']])
        # pair_data_unseen     = np.zeros([len(unseen_pairs), kwargs['complex_dim']])
        pair_data_unseen_att = np.zeros([len(unseen_pairs), kwargs['complex_dim']])
        pair_data_unseen_obj = np.zeros([len(unseen_pairs), kwargs['complex_dim']])
        
        # test_data     = np.zeros([test_sample_num, kwargs['complex_dim']])
        test_data_att = np.zeros([test_sample_num, kwargs['complex_dim']])
        test_data_obj = np.zeros([test_sample_num, kwargs['complex_dim']])
        i = 0
        for _, data in tqdm(enumerate(dataloader), desc='GT Feature', total=len(dataloader), disable=args.no_pbar, **tqdm_commons):
            if args.parallel:
                output = model.module.forward(data[0].float(), None, None, status='eval', vis_backbone=kwargs['vis_backbone'])
            else:
                output = model.forward(data[0].float(), None, None, status='eval', vis_backbone=kwargs['vis_backbone'])
            feat_tmp = output['im_feat']
            # test_data[i:i + feat_tmp.shape[0], :]     = output['im_feat'].detach().cpu().numpy()
            test_data_att[i:i + feat_tmp.shape[0], :] = output['im_att_feat'].detach().cpu().numpy()
            test_data_obj[i:i + feat_tmp.shape[0], :] = output['im_obj_feat'].detach().cpu().numpy()
            # test_data_residue[i:i + dataloader.batch_size, :] = model.get_residue(data[0].cuda()).detach().cpu().numpy()
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
            # tmp = output['lin_feat_recs']
            # pair_data_unseen[i, :]     = output['lin_feat_recs'].detach().cpu().numpy()
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
            # tmp = output['lin_feat_recs']
            # pair_data_seen[i, :]     = output['lin_feat_recs'].detach().cpu().numpy()
            pair_data_seen_att[i, :] = output['lin_att_values'].detach().cpu().numpy()
            pair_data_seen_obj[i, :] = output['lin_obj_values'].detach().cpu().numpy()
            if args.debug_mode:
                break
        # pair_data_seen[0:len(unseen_pairs), :] = pair_data_unseen

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
        
   
    seen_acc = float(correct_seen.max())
    unseen_acc = float(correct_unseen.max())
    test_info = {
        'phase':    phase,
        'auc':      float(auc),
        'seen_acc':   seen_acc,
        'unseen_acc': unseen_acc,
        'overall_acc': (total_seen * seen_acc + total_unseen * unseen_acc) / (total_seen + total_unseen),
        'h_mean':   float(h_mean(correct_unseen, correct_seen).max())
    }
    
    return test_info


def split_meta(*args, meta_samples=0.9, meta_inclusive=False):
    a_label_p, o_label_p = args
    all_indices = list(range(len(a_label_p)))
    meta_idx = random.sample(all_indices, int(len(all_indices)*meta_samples))
    if meta_inclusive:
        train_idx = all_indices
    else:
        train_idx = list(set(all_indices) - set(meta_idx))
    return train_idx, meta_idx


def train_step(model, optimizer, data_loader, loss_func=loss_separate, meta_loss_func=None, meta_optimizer=None, device=torch.device('cuda'), args=None, **kwargs):
    model.train()
    train_info = {}
    train_loss_avg = RunningAverage(len(data_loader))
    train_loss_meta_avg = RunningAverage(len(data_loader))

    loss_weights = kwargs['loss_weights']

    t = tqdm(data_loader, disable=args.no_pbar, **tqdm_commons)
    should_print = True
    correct = att_correct = obj_correct = acc_att = acc_obj = total_count = 0
    with torch.autograd.set_detect_anomaly(args.debug_mode):
        for i, data in enumerate(t):

            img_p, a_label_p, o_label_p = data[0], data[3], data[4]
            img_pn, att_idx_pn, obj_idx_pn = data[0+7], data[3+7], data[4+7]
            img_np, att_idx_np, obj_idx_np = data[0+14], data[3+14], data[4+14]

            if meta_optimizer is not None:
                train_idx, meta_idx = split_meta(a_label_p, o_label_p, meta_samples=args.meta_samples, meta_inclusive=True)
            else:
                train_idx = list(range(len(a_label_p)))

            loss = torch.Tensor([0])
            rand_a = np.random.rand()
            rand_o = np.random.rand()
            if len(train_idx) > 0:
                output_pp = model.forward(img_p[train_idx], a_label_p[train_idx], o_label_p[train_idx], mask_target=False, vis_backbone=kwargs['vis_backbone'])
                att_preds = torch.argmax(torch.softmax(output_pp['im_att_logits'], -1), -1)
                obj_preds = torch.argmax(torch.softmax(output_pp['im_obj_logits'], -1), -1)
                att_correct = (att_preds == a_label_p.to(att_preds.device))
                obj_correct = (obj_preds == o_label_p.to(att_preds.device))
                correct += int((att_correct & obj_correct).sum())
                acc_att += int(att_correct.sum())
                acc_obj += int(obj_correct.sum())
                total_count += data[0].shape[0]
                negs = []
                
                output_pn = model.forward(img_pn[train_idx], att_idx_pn[train_idx], obj_idx_pn[train_idx], mask_target=False, vis_backbone=kwargs['vis_backbone'])  # pos att, neg obj
                output_np = model.forward(img_np[train_idx], att_idx_np[train_idx], obj_idx_np[train_idx], mask_target=False, vis_backbone=kwargs['vis_backbone'])  # neg att, pos obj

                negs.append((output_pn, output_np))
                
                losses = loss_func(output_pp, negs, rand_a=rand_a, rand_o=rand_o, loss_weights=loss_weights, should_print=should_print, rank_margin=args.rank_margin, args=args)
                loss = 0
                for _, v in losses.items():
                    loss += v
                t.set_description(''.join([f'{k}={round(v.item(),3)} ' for k, v in losses.items()]))

                if loss != loss:
                    logging.getLogger(myself()).critical('Training aborted because loss becomes NaN.')
                    raise ValueError
                
                if loss != 0:
                    model.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

            # extra generalization training
            loss_meta = torch.Tensor([0])
            if meta_optimizer is not None:
                output_pp = model.forward(img_p[meta_idx], a_label_p[meta_idx], o_label_p[meta_idx], mask_target=False, vis_backbone=kwargs['vis_backbone'], ignore_img=True)
                output_pp.update({'masked': model.forward(img_p[meta_idx], a_label_p[meta_idx], o_label_p[meta_idx], mask_target=True, vis_backbone=kwargs['vis_backbone'])})
                
                output_pn = {}
                output_pn.update({'masked': model.forward(img_pn[meta_idx], att_idx_pn[meta_idx], obj_idx_pn[meta_idx], mask_target=True, vis_backbone=kwargs['vis_backbone'])})
                
                output_np = {}
                output_np.update({'masked': model.forward(img_np[meta_idx], att_idx_np[meta_idx], obj_idx_np[meta_idx], mask_target=True, vis_backbone=kwargs['vis_backbone'])})
                
                losses_meta = meta_loss_func(output_pp, [(output_pn, output_np)], rand_a=rand_a, rand_o=rand_o, loss_weights=loss_weights, should_print=should_print, rank_margin=args.rank_margin, args=args)
                loss_meta = 0
                for _, v in losses_meta.items():
                    loss_meta += v
                if loss_meta != loss_meta:
                    logging.getLogger(myself()).critical('Training aborted because loss_meta becomes NaN.')
                    raise ValueError
                model.zero_grad()
                loss_meta.backward(retain_graph=True)
                meta_optimizer.step()
        
            if loss != 0:
                train_info = {
                    'phase': 'train',
                    'loss': train_loss_avg.add(loss.item()),
                    'loss_meta': train_loss_meta_avg.add(loss_meta.item()),
                    'acc': correct / total_count,
                    'acc_att': acc_att / total_count,
                    'acc_obj': acc_obj / total_count
                }
            else:
                train_info = {
                    'phase': 'train',
                    'loss': 0,
                    'loss_meta': train_loss_meta_avg.add(loss_meta.item()),
                    'acc': correct / total_count,
                    'acc_att': acc_att / total_count,
                    'acc_obj': acc_obj / total_count
                }

            if args.debug_mode:
                break
            should_print = False

    return train_info


if __name__ == '__main__':
    args = init(user_param=params)

    obj_class_num    = {'ut-zap50k': 12,  'mitstates': 245}
    att_class_num    = {'ut-zap50k': 16,  'mitstates': 115}
    obj_encode_dims  = {'ut-zap50k': 300, 'mitstates': 300}
    att_encode_dims  = {'ut-zap50k': 300, 'mitstates': 300}

    loss_weights = json.loads(args.loss_weights)

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

    device = torch.device('cuda')
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
    
    model = SepMix(**model_config).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_schdlr = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.sch_milestones, gamma=0.1, last_epoch=args.start_epoch-1)
    vis_backbone = None if args.pre_feat else torch.nn.Sequential(*list(resnet18(pretrained=True).children())[:-1]).cuda().eval()
    
    val = val_separate  # val_separate val_distance
    loss_func = loss_separate  # loss_separate loss_dist
    meta_loss_func = loss_meta  # loss_meta loss_meta_dist
    meta_optimizer = optimizer if args.meta_samples > 0 else None

    model, optimizer, lr_schdlr = prepare_train(model, optimizer, lr_schdlr, args)

    writer = SummaryWriter(log_dir=args.summary_to)
    if not args.test_only:
        shutil.copy(osp.join('model', 'SepMask.py'), osp.join(args.save_model_to, args.model_id, 'SepMask.py'))
        shutil.copy(osp.join('.', 'train.py'), osp.join(args.save_model_to, args.model_id, 'train.py'))

        ss = ShouldSaveModel(init_step=args.start_epoch-1)
        es = EarlyStop(patience=args.patience)

        for epoch in range(args.start_epoch, args.max_epoch):
            logging.getLogger(myself()).info("-"*10 + f" Epoch {epoch} starts. " + "-"*10)  # for timing
            with elapsed_timer() as elapsed:
                train_info = train_step(model, optimizer, train_dataloader, loss_func=loss_func, meta_loss_func=meta_loss_func, meta_optimizer=meta_optimizer, loss_weights=loss_weights, device=device, vis_backbone=vis_backbone, args=args)
            logging.getLogger(myself()).info(f"Epoch {epoch} finished. Elapsed={elapsed():.2f}s.")  # for timing
            logging.getLogger(myself()).info(
                f"Epoch {epoch}, "
                f"{train_info}"
            )

            val_info = val(model, val_dataloader, phase='val', device=device, complex_dim=args.latent_dims, vis_backbone=vis_backbone, args=args)
            logging.getLogger(myself()).info(
                f"Epoch {epoch}, "
                f"{val_info}"
            )

            lr_schdlr.step()

            states_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_schdlr.state_dict(),
                'checkpoint_epoch': epoch,
                'initial_lr': args.lr
            }
            if ss.step(loss=1e3, acc=val_info['auc'], criterion=lambda x1, x2: x2):
                save_checkpoint(
                    f'{args.save_model_to}/{args.model_id}/best.state', **states_dict)
            save_checkpoint(
                    f'{args.save_model_to}/{args.model_id}/latest.state', **states_dict)
            
            if es.step(loss=1e3, acc=val_info['auc'], criterion=lambda x1, x2: x2):
                break

            if args.debug_mode:
                break
        
        logging.getLogger(myself()).info('Training ended.')
        states = load_checkpoint(f'{args.save_model_to}/{args.model_id}/best.state', state_dict_to_load=['model', 'checkpoint_epoch'])
        best_epoch = states['checkpoint_epoch']
        model.load_state_dict(states['model'])
        
        test_info = val(model, test_dataloader, phase='test', device=device, complex_dim=args.latent_dims, vis_backbone=vis_backbone, args=args)
        logging.getLogger(myself()).info(
                f"Best model at epoch {best_epoch}, "
                f"{test_info}"
        )
    elif args.test_only:
        best_epoch = 0
        pretrained = torch.load(os.path.join(args.model_dir, 'best.state'))
        model.load_state_dict(pretrained['model'])
        for i in range(3):
            print("step: ", i+1)
            test_info = val(model, test_dataloader, topk=i+1, phase='test', device=device, complex_dim=args.latent_dims, vis_backbone=vis_backbone, args=args)
            print("test: {}".format(i+1))
            print(test_info)
            print()
            test_info = val(model, val_dataloader, topk=i+1, phase='test', device=device, complex_dim=args.latent_dims, vis_backbone=vis_backbone, args=args)
            print("val: {}".format(i+1))
            print(test_info)
            print('\n\n')

    writer.close()

