import itertools
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from scipy.sparse import diags


VIS_BACKBONE_FEAT_DIM = 512

class Discriminator(nn.Module):
    def __init__(self, input_dims=512, hidden_dims=512, output_dims=2):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dims, hidden_dims)
        self.rl1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)
        self.lsm = nn.LogSoftmax()


    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.rl1(x1)
        x3 = self.fc2(x2)
        x4 = self.lsm(x3)
        return x4
        
def _to_onehot(labels, num_classes):
    oh = torch.zeros(labels.shape[0], num_classes).cuda()
    oh.scatter_(1, labels.unsqueeze(1).long(), 1)
    return oh


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = mx.sum(-1)
    r_inv = (1 / rowsum)
    r_inv[torch.isinf(r_inv)] = 0
    if len(mx.shape) == 2:
        r_mat_inv = torch.diag(r_inv.flatten())
        mx = r_mat_inv @ mx
    else:
        r_mat_inv = torch.diag_embed(r_inv)
        mx = r_mat_inv.bmm(mx)
    return mx


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    Re-implemented using Conv1d to support batch operation.
    """

    def __init__(self, in_features, out_features, bias=True, groups=1, adj=None, num_weights=1, **kwargs):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adj = adj
        self.groups = groups

        self.weight = nn.Parameter(torch.FloatTensor(num_weights, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj=None, should_normalize=True):
        if len(x.shape) == 3:
            support = (x.unsqueeze(-2) @ self.weight[None, :, :].repeat(x.shape[0], 1, 1, 1)).squeeze(-2)
        else:
            support = (x[:, None, :] @ self.weight).squeeze(-2)

        if should_normalize:
            adj = normalize(adj)
        adj = adj.cuda()
        output = adj @ support

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ', ' + f'Groups={self.groups}' + ')'


class VisTransformer(nn.Module):
    def __init__(self, input_dim=960, output_dim=960, hidden_units=1024, dropout=0.0):
        super(VisTransformer, self).__init__()

        self.dropout = dropout
        self.in_layer  = nn.Linear(input_dim, hidden_units)
        self.out_layer = nn.Linear(hidden_units, output_dim)

    def forward(self, x):
        x_ = F.dropout(self.in_layer(x), p=self.dropout, training=self.training)
        x_ = F.leaky_relu(x_)

        x_ = F.dropout(self.out_layer(x_), p=self.dropout, training=self.training)
        x_ = F.leaky_relu(x_)

        return x_

class GaussianResidualGenerator(nn.Module):
    def __init__(self, dim=512):
        super(GaussianResidualGenerator, self).__init__()
        self.dim = dim
        self.mean = nn.Parameter(torch.randn(dim).cuda())
        self.covariance = nn.Parameter(torch.eye(dim).cuda())
        self.dist = torch.distributions.multivariate_normal.MultivariateNormal(self.mean, self.covariance)
    
    def forward(self, x, status):
        if status == 'train':
            ret = self.dist.rsample()
            ret = ret.to(x.device)
            return ret
        elif status == 'eval':
            ret = self.dist.mean
            ret = ret.to(x.device)
            return ret


class SepMix(nn.Module):
    def __init__(self, **kwargs):
        super(SepMix, self).__init__()

        self.args = kwargs['args']
        self.dropout = kwargs['dropout']
        self.primitive_dim = kwargs['primitive_dim']
        self.complex_dim = kwargs['complex_dim']
        self.att_encodes = kwargs['att_encodes']
        self.obj_encodes = kwargs['obj_encodes']
        self.att_class_num = kwargs['att_class_num']
        self.obj_class_num = kwargs['obj_class_num']
        self.seen_mask = kwargs['seen_mask']
        self.kq_dim = kwargs.get('kq_dim', 300)
        self.vis_mem_blocks = kwargs.get('vis_mem_blocks', 64)
        self.code_dim = 128

        self.vis_transform = nn.Sequential(
            nn.Linear(VIS_BACKBONE_FEAT_DIM, 1024),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, self.complex_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.complex_dim)
        )
        self.residual_generator = GaussianResidualGenerator()
        if self.att_class_num > 16:
            self.lin_att_gcn = GraphConvolution(in_features=self.primitive_dim, out_features=self.complex_dim, num_weights=self.att_class_num+self.obj_class_num)
            self.lin_obj_gcn = GraphConvolution(in_features=self.primitive_dim, out_features=self.complex_dim, num_weights=self.att_class_num+self.obj_class_num)
        else:
            self.lin_att_gcn = GraphConvolution(in_features=self.primitive_dim, out_features=self.complex_dim, num_weights=self.att_class_num+self.obj_class_num)
            self.lin_obj_gcn = GraphConvolution(in_features=self.primitive_dim, out_features=self.complex_dim, num_weights=self.att_class_num+self.obj_class_num)
        
        self.lin_att_key = nn.Parameter(torch.Tensor(kwargs['att_class_num'], self.kq_dim))
        self.lin_obj_key = nn.Parameter(torch.Tensor(kwargs['obj_class_num'], self.kq_dim))
        torch.nn.init.normal_(self.lin_att_key)
        torch.nn.init.normal_(self.lin_obj_key)
        self.lin_att_query = nn.Parameter(torch.Tensor(kwargs['att_class_num'], self.kq_dim))
        self.lin_obj_query = nn.Parameter(torch.Tensor(kwargs['obj_class_num'], self.kq_dim))
        torch.nn.init.normal_(self.lin_att_query)
        torch.nn.init.normal_(self.lin_obj_query)

        self.att_query_transformer = nn.Sequential(
            nn.Linear(self.kq_dim, self.kq_dim),
            nn.Tanh())
        self.obj_query_transformer = nn.Sequential(
            nn.Linear(self.kq_dim, self.kq_dim),
            nn.Tanh())
        self.att_key_transformer = nn.Sequential(
            # nn.Linear(self.kq_dim, self.kq_dim),
            nn.Tanh())
        self.obj_key_transformer = nn.Sequential(
            # nn.Linear(self.kq_dim, self.kq_dim),
            nn.Tanh())

        self.lin_att_values = nn.Parameter(torch.Tensor(kwargs['att_class_num'], self.primitive_dim))
        self.lin_obj_values = nn.Parameter(torch.Tensor(kwargs['obj_class_num'], self.primitive_dim))
        torch.nn.init.normal_(self.lin_att_values)
        torch.nn.init.normal_(self.lin_obj_values)

        # visual query, key, values
        self.vis_att_transformer = nn.Sequential(
            nn.Linear(self.complex_dim, self.complex_dim),
            nn.LeakyReLU(),
        )
        self.vis_obj_transformer = nn.Sequential(
            nn.Linear(self.complex_dim, self.complex_dim),
            nn.LeakyReLU(),
        )

        self.att_cls = nn.Sequential(
            nn.Linear(self.complex_dim, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, self.att_class_num),
        )

        self.obj_cls = nn.Sequential(
            nn.Linear(self.complex_dim, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, self.obj_class_num),
        )

        self.gcn_output_merge = nn.Sequential(
            nn.Linear(2*self.complex_dim, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, self.complex_dim),
            nn.LeakyReLU()
        )

    def forward(self, im_feat, att_idx, obj_idx, status='train', mask_target=True, **kwargs):
        assert att_idx is not None or im_feat is not None, "Parameter error."
        output = {}
        ignore_img = kwargs.get('ignore_img', False)

        if att_idx is not None:
            assert self.lin_att_key.shape[0] == self.att_class_num
            batch_size = len(att_idx)
            att_idx = att_idx.cuda()
            obj_idx = obj_idx.cuda()

            cat_key = torch.cat([self.lin_att_key, self.lin_obj_key], 0)
            cat_values = torch.cat([self.lin_att_values, self.lin_obj_values], dim=0)[None, :, :]

            A_cpl_att_ = self.att_key_transformer(self.lin_att_query[att_idx]) @ self.att_query_transformer(cat_key).t()  # / np.sqrt(self.att_query.shape[1])
            A_cpl_obj_ = self.obj_key_transformer(self.lin_obj_query[obj_idx]) @ self.obj_query_transformer(cat_key).t()  # / np.sqrt(self.obj_query.shape[1])

            if mask_target:
                seen_mask = self.seen_mask.cuda()
                A_cpl_att_1_ = A_cpl_att_.masked_fill(~seen_mask[att_idx].cuda(), float('-inf'))
                A_cpl_obj_1_ = A_cpl_obj_.masked_fill(~seen_mask[self.att_class_num+obj_idx].cuda(), float('-inf'))
                A_cpl_att_1_[torch.arange(batch_size), self.att_class_num+obj_idx] = float('-inf')
                A_cpl_obj_1_[torch.arange(batch_size), att_idx] = float('-inf')
            else:
                A_cpl_att_1_ = A_cpl_att_
                A_cpl_obj_1_ = A_cpl_obj_

            A_cpl_att_1_ = torch.cat((torch.softmax(A_cpl_att_1_[:, :self.att_class_num], -1), torch.softmax(A_cpl_att_1_[:, self.att_class_num:], -1)), -1)
            A_cpl_att_1_ = A_cpl_att_1_.masked_fill(torch.isnan(A_cpl_att_1_), 0.0)
            A_cpl_obj_1_ = torch.cat((torch.softmax(A_cpl_obj_1_[:, :self.att_class_num], -1), torch.softmax(A_cpl_obj_1_[:, self.att_class_num:], -1)), -1)
            A_cpl_obj_1_ = A_cpl_obj_1_.masked_fill(torch.isnan(A_cpl_obj_1_), 0.0)

            A_cpl_att_1 = torch.zeros(
                (self.att_class_num+self.obj_class_num, self.att_class_num+self.obj_class_num))[None, :, :].repeat(batch_size, 1, 1).cuda()
            A_cpl_obj_1 = torch.zeros(
                (self.att_class_num+self.obj_class_num, self.att_class_num+self.obj_class_num))[None, :, :].repeat(batch_size, 1, 1).cuda()
            A_cpl_att_1[torch.arange(batch_size), att_idx, :] = A_cpl_att_1_[torch.arange(batch_size)]
            A_cpl_obj_1[torch.arange(batch_size), self.att_class_num+obj_idx, :] = A_cpl_obj_1_[torch.arange(batch_size)]

            lin_att_values_ = F.leaky_relu(self.lin_att_gcn.forward(cat_values, adj=A_cpl_att_1, should_normalize=False))
            lin_obj_values_ = F.leaky_relu(self.lin_obj_gcn.forward(cat_values, adj=A_cpl_obj_1, should_normalize=False))
            lin_att_values = lin_att_values_[torch.arange(batch_size), att_idx]
            lin_obj_values = lin_obj_values_[torch.arange(batch_size), self.att_class_num+obj_idx]

            lin_feat_recs = F.leaky_relu(self.gcn_output_merge(torch.cat([lin_att_values, lin_obj_values], -1)))

            lin_att_logits = self.att_cls(lin_att_values)
            lin_obj_logits = self.obj_cls(lin_obj_values)

            output.update({
                'lin_feat_recs': lin_feat_recs,
                'lin_att_logits': lin_att_logits, 'lin_obj_logits': lin_obj_logits,
                'lin_att_values': lin_att_values, 'lin_obj_values': lin_obj_values,
                'att_idx': att_idx, 'obj_idx': obj_idx})

        if (im_feat is not None) and (not ignore_img):
            im_feat = im_feat.cuda()
            if len(im_feat.shape) == 2:
                im_feat_transformed = self.vis_transform(im_feat)
                
            else:
                assert 'vis_backbone' in kwargs
                with torch.no_grad():
                    im_feat = kwargs['vis_backbone'](im_feat).squeeze()
                    im_feat_transformed = self.vis_transform(im_feat)
            residual = self.residual_generator(im_feat_transformed, status)
            im_feat_reduced = im_feat_transformed - residual
            im_feat1 = im_feat_reduced
            im_feat2 = im_feat_reduced
            im_att_feat = self.vis_att_transformer(im_feat1)
            im_obj_feat = self.vis_obj_transformer(im_feat2)

            im_att_logits = self.att_cls(im_att_feat)
            im_obj_logits = self.obj_cls(im_obj_feat)

            output.update({
                'im_feat': im_feat,
                'im_att_feat': im_att_feat, 'im_obj_feat': im_obj_feat,
                'im_att_logits': im_att_logits, 'im_obj_logits': im_obj_logits,
                'im_att_fake_logits': None, 'im_obj_fake_logits': None})

        return output