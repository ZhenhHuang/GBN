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
            nn.Linear(in_dim, hid_dim, bias=bias),
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
    def __init__(self, in_dim, hid_dim, out_dim, bias,
                 act='gelu', drop=0.3, add_self_loop=False):
        super().__init__()
        self.add_self_loop = add_self_loop
        self.lin = nn.Sequential(
            nn.Linear(in_dim, hid_dim, bias=bias),
            nn.Dropout(drop)
        )
        self.alpha = nn.Sequential(nn.Linear(hid_dim, hid_dim, bias=bias),
                                   nn.Dropout(drop),
                                   nn.ReLU(),
                                   nn.Linear(hid_dim, hid_dim, bias=bias),
                                   nn.Softplus(),
                                   )
        self.beta = nn.Sequential(nn.Linear(hid_dim, hid_dim, bias=bias),
                                  nn.Dropout(drop),
                                  nn.ReLU(),
                                  nn.Linear(hid_dim, hid_dim, bias=bias),
                                  nn.Softplus(),
                                  )
        self.gamma = nn.Sequential(nn.Linear(in_dim, in_dim, bias=bias),
                                   nn.Dropout(drop),
                                   ActivateModule(act),
                                   nn.Linear(hid_dim, hid_dim, bias=bias)
                                   )
        self.fc = FeedForwardLayer(hid_dim, hid_dim, out_dim, bias, act, drop)
        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, x, edge_index, degree):
        """
        x : N * D
        edge_index: 2 * E
        degree: N,1
        """
        x = self.lin(x)
        x_res = self.norm(x)
        alpha = self.alpha(x)
        beta = self.beta(x)
        gamma = self.gamma(x)
        x = (beta * self.aggregate(x, edge_index) + gamma) / (alpha + beta * degree)
        x = self.fc(x) + x_res
        return x

    @staticmethod
    def aggregate(x, edge_index):
        num_nodes = x.shape[0]
        src, dst = edge_index[0], edge_index[1]
        x = scatter_sum(x[src], dst, dim=0, dim_size=num_nodes)
        return x