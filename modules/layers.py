import torch.nn as nn
import torch
from torch_scatter import scatter_sum
from torch_geometric.utils import normalize_edge_index
from utils.train_utils import ActivateModule, NormModule
import torch.nn.functional as F


class FeedForwardLayer(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, bias, act='gelu', drop=0.3):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim, hid_dim, bias=bias),
            nn.Dropout(drop),
            ActivateModule(act),
            nn.Linear(hid_dim, out_dim, bias=bias),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class GCNLayer(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, bias,
                 act='gelu', drop=0.3, add_self_loop=False):
        super().__init__()
        self.fc = FeedForwardLayer(in_dim, hid_dim, out_dim, bias, act, drop)
        self.add_self_loop = add_self_loop

    def forward(self, x, edge_index):
        x = self.aggregate(x, edge_index, self.add_self_loop)
        x = self.fc(x)
        return x

    @staticmethod
    def aggregate(x, edge_index, add_self_loop):
        num_nodes = x.shape[0]
        edge_index, edge_weight = normalize_edge_index(edge_index, num_nodes,
                                                       add_self_loops=add_self_loop)
        src, dst = edge_index[0], edge_index[1]
        x = scatter_sum(edge_weight.unsqueeze(1) * x[src], index=dst, dim=0, dim_size=num_nodes)
        return x


class BoundaryConvLayer(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, bias, act='gelu', drop=0.3, norm='ln',
                 rate=None, gamma=None):
        super().__init__()
        self.rate = nn.Sequential(nn.Linear(in_dim, hid_dim, bias=bias),
                                  nn.Dropout(drop),
                                  ActivateModule(act),
                                  nn.Linear(hid_dim, hid_dim, bias=bias),
                                  NormModule(norm, hid_dim)
                                  ) if rate is None else rate
        self.gamma = nn.Sequential(nn.Linear(in_dim, hid_dim, bias=bias),
                                   nn.Dropout(drop),
                                   ActivateModule(act),
                                   nn.Linear(hid_dim, hid_dim, bias=bias),
                                   NormModule(norm, hid_dim)
                                   ) if rate is None else gamma
        self.fc = FeedForwardLayer(hid_dim, hid_dim, out_dim, bias, act, drop)

    def forward(self, xt, x0, edge_index, ind_bd):
        """
        x : N * D
        edge_index: 2 * E
        degrees: N
        """
        rate = self.rate(xt)
        gamma = self.gamma(x0)
        ind_int = 1 - ind_bd
        p_deg = ind_bd * self.aggregate(torch.ones_like(ind_bd), edge_index) + (ind_int - ind_bd) * self.aggregate(
            ind_int, edge_index) + 1
        in_x = ind_int / torch.sqrt(p_deg) * self.aggregate((1 + ind_int) * xt / torch.sqrt(p_deg), edge_index,
                                                            src2dst=True)
        out_x = rate * ind_bd / torch.sqrt(p_deg) * self.aggregate(ind_int * xt / torch.sqrt(p_deg), edge_index,
                                                                   src2dst=True)
        x = self.fc(in_x + out_x) + gamma
        return x

    @staticmethod
    def aggregate(x, edge_index, src2dst: bool = True):
        num_nodes = x.shape[0]
        src, dst = edge_index[0], edge_index[1]
        if src2dst:
            x = scatter_sum(x[src], dst, dim=0, dim_size=num_nodes)
        else:
            x = scatter_sum(x[dst], src, dim=0, dim_size=num_nodes)
        return x