import torch.nn as nn
import torch.nn.functional as F
from utils.train_utils import ActivateModule, NormModule
from modules.layers import BoundaryConvLayer, GCNLayer, FeedForwardLayer
from torch_geometric.utils import add_self_loops


class BoundaryGCN(nn.Module):
    def __init__(self, n_layers, in_dim, hid_dim, embed_dim, out_dim,
                 bias, input_act, act, drop=0.3, norm='ln',
                 add_self_loop=True, tau=0.1, layer_wise=True):
        super().__init__()
        self.input_lin = nn.Sequential(nn.Linear(in_dim, embed_dim, bias=bias),
                                       nn.Dropout(drop),
                                       ActivateModule(input_act))
        self.rate = nn.Sequential(nn.Linear(embed_dim, hid_dim, bias=bias),
                                  nn.Dropout(drop),
                                  ActivateModule(act),
                                  nn.Linear(hid_dim, hid_dim, bias=bias),
                                  NormModule(norm, hid_dim)
                                  ) if layer_wise else None
        self.gamma = nn.Sequential(nn.Linear(embed_dim, hid_dim, bias=bias),
                                   nn.Dropout(drop),
                                   ActivateModule(act),
                                   nn.Linear(hid_dim, hid_dim, bias=bias),
                                   NormModule(norm, hid_dim)
                                   ) if layer_wise else None
        self.layers = nn.ModuleList([BoundaryConvLayer(embed_dim, hid_dim, embed_dim,
                                                       bias, act, drop, norm,
                                                       self.rate, self.gamma) for _ in range(n_layers)])
        self.out_norm = NormModule(norm, hid_dim)
        self.out_lin = nn.Linear(embed_dim, out_dim, bias=bias)
        self.drop = nn.Dropout(drop)
        self.add_self_loop = add_self_loop
        self.ind_layer = GCNLayer(embed_dim, hid_dim, 1, bias=bias, drop=drop)
        self.tau = tau

    def forward(self, data):
        x = data.x
        x0 = self.input_lin(x)
        xt = x0
        edge_index = add_self_loops(data.edge_index)[0] if self.add_self_loop else data.edge_index
        ind_bd = F.logsigmoid(self.ind_layer(x0, edge_index) / self.tau).exp()
        for layer in self.layers:
            xt = layer(xt, x0, edge_index, ind_bd)
        xt = self.out_norm(xt)
        x = self.out_lin(xt)
        return x


class GCN(nn.Module):
    def __init__(self, n_layers, in_dim, hid_dim, embed_dim, out_dim,
                 bias, input_act, act, drop=0.3, norm='ln'):
        super().__init__()
        self.input_lin = nn.Sequential(nn.Linear(in_dim, embed_dim, bias=bias),
                                       nn.Dropout(drop),
                                       ActivateModule(input_act))
        self.layers = nn.ModuleList([GCNLayer(embed_dim, hid_dim, embed_dim,
                                              bias, act, drop) for _ in range(n_layers)])
        self.out_norm = NormModule(norm, hid_dim)
        self.out_lin = nn.Linear(embed_dim, out_dim, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        x = self.input_lin(x)
        for layer in self.layers:
            x = layer(x, edge_index)
        x = self.out_norm(x)
        x = self.out_lin(x)
        return x
