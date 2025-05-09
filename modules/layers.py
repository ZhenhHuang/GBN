import torch.nn as nn
import torch
from torch_scatter import scatter_sum
from torch_geometric.utils import normalize_edge_index
from utils.train_utils import ActivateModule
import torch.nn.functional as F


class FeedForwardLayer(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, bias, act='gelu', drop=0.3):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim, hid_dim, bias=bias),
            nn.Dropout(drop),
            ActivateModule(act),
            nn.Linear(in_dim, out_dim, bias=bias),
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
    def __init__(self, in_dim, hid_dim, out_dim, bias, act='gelu', drop=0.3, norm='ln'):
        super().__init__()
        self.lin = nn.Sequential(nn.Linear(in_dim, hid_dim, bias=bias),
                                 nn.Dropout(drop))
        self.rate = nn.Sequential(nn.Linear(hid_dim, hid_dim, bias=bias),
                                  nn.Dropout(drop),
                                  ActivateModule(act),
                                  nn.Linear(hid_dim, hid_dim, bias=bias),
                                  nn.LayerNorm(hid_dim))
        self.dir_bound = nn.Sequential(nn.Linear(hid_dim, hid_dim, bias=bias),
                                       nn.Dropout(drop),
                                       ActivateModule(act),
                                       nn.Linear(hid_dim, hid_dim, bias=bias),
                                       nn.LayerNorm(hid_dim))
        self.rob_bound = nn.Sequential(nn.Linear(hid_dim, hid_dim, bias=bias),
                                       nn.Dropout(drop),
                                       ActivateModule(act),
                                       nn.Linear(hid_dim, hid_dim, bias=bias),
                                       nn.LayerNorm(hid_dim))
        self.fc = FeedForwardLayer(hid_dim, hid_dim, out_dim, bias, act, drop)
        self.norm = nn.LayerNorm(hid_dim) if norm == 'ln' else nn.BatchNorm1d(hid_dim)

    def forward(self, x, edge_index, degree):
        """
        x : N * D
        edge_index: 2 * E
        degrees: N
        """
        x = self.lin(x)
        x_res = self.norm(x)
        alpha = self.dir_bound(x)
        beta = self.rate(x)
        gamma = self.rob_bound(x)
        in_x = alpha * self.aggregate(x, edge_index, src2dst=True)
        out_x = self.aggregate(beta * x, edge_index, src2dst=False)
        x = in_x + gamma + out_x
        x = self.fc(x) + x_res
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