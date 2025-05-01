import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum
from torch_geometric.utils import add_self_loops
from utils.train_utils import ActivateModule


EPS = 1e-4


class BoundaryConvLayer(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, bias, act='gelu', drop=0.3, norm='ln'):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, hid_dim, bias=bias),
                                nn.Dropout(drop),
                                ActivateModule(act),
                                nn.Linear(hid_dim, out_dim),
                                nn.Dropout(drop))
        self.rate = nn.Sequential(nn.Linear(in_dim, in_dim, bias=bias),
                                  nn.Softplus(),
                                  nn.Dropout(drop))
        self.rob_bound = nn.Sequential(nn.Linear(in_dim, hid_dim, bias=bias),
                                       nn.Dropout(drop),
                                       nn.Softplus(),
                                       nn.Linear(hid_dim, in_dim, bias=bias),
                                       nn.LayerNorm(in_dim))
        self.norm = nn.LayerNorm(in_dim) if norm == 'ln' else nn.BatchNorm1d(in_dim)

    def forward(self, x, edge_index, degree):
        """
        x : N * D
        edge_index: 2 * E
        degrees: N
        """
        x_res = self.norm(x)
        rate = self.rate(x)
        gamma = self.rob_bound(x)
        x = self.aggregate(x, edge_index)
        x = (rate * x + gamma) / (1 + rate * degree.unsqueeze(1) + EPS)
        x = self.fc(x) + x_res
        return x

    def aggregate(self, x, edge_index):
        num_nodes = x.shape[0]
        src, dst = edge_index[0], edge_index[1]
        x = scatter_sum(x[src], dst, dim=0, dim_size=num_nodes)
        return x


class BoundaryGCN(nn.Module):
    def __init__(self, n_layers, in_dim, hid_dim, embed_dim, out_dim,
                 bias, input_act, act, drop=0.3, norm='ln', add_self_loop=True):
        super().__init__()
        self.input_lin = nn.Sequential(nn.Linear(in_dim, embed_dim, bias=bias),
                                       nn.Dropout(drop),
                                       ActivateModule(input_act))
        self.layers = nn.ModuleList([BoundaryConvLayer(embed_dim, hid_dim, embed_dim, bias, act, drop, norm) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(embed_dim) if norm == 'ln' else nn.BatchNorm1d(embed_dim)
        self.out_lin = nn.Linear(embed_dim, out_dim, bias=bias)
        self.drop = nn.Dropout(drop)
        self.add_self_loop = add_self_loop

    def forward(self, data):
        x = data.x
        x = self.input_lin(x)
        x = self.drop(x)
        edge_index = add_self_loops(data.edge_index)[0] if self.add_self_loop else data.edge_index
        degree = data.degree
        for layer in self.layers:
            x = layer(x, edge_index, degree)
        x = self.out_norm(x)
        x = self.out_lin(x)
        return x