import math

import numpy as np
from scipy.sparse import diags
import torch
from torch.nn import Module, Parameter


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    Re-implemented using Conv1d to support batch operation.
    """

    def __init__(self, in_features, out_features, bias=True, groups=1, adj=None, **kwargs):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adj = adj
        self.groups = groups
        # TODO: to support different weights for different nodes (maybe unfold -> matmul -> fold trick)
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj=None, should_normalize=True):
        support = torch.mm(x, self.weight)
        if should_normalize:
            adj = torch.Tensor(normalize(adj)).to(self.weight.device)
        output = torch.matmul(adj, support)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ', ' + f'Groups={self.groups}' + ')'


if __name__ == "__main__":
    gc = GraphConvolution(in_features=512, out_features=32, bias=True)
    x = torch.randn((32, 512))
    adj = torch.randn((32, 32))
    print(gc(x, adj).shape)