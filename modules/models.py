import torch.nn as nn
import torch
from utils.train_utils import ActivateModule
from modules.layers import BoundaryConvLayer
from torch_geometric.utils import add_self_loops


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
        edge_index = add_self_loops(data.edge_index)[0] if self.add_self_loop else data.edge_index
        degree = data.degree + 1 if self.add_self_loop else data.degree
        for layer in self.layers:
            x = layer(x, edge_index, degree)
        x = self.out_norm(x)
        x = self.out_lin(x)
        return x