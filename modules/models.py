import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum
from torch_geometric.nn import GraphNorm


EPS = 1e-4


class BoundaryConvLayer(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, bias, act=None, drop=0.3):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=bias)
        self.act = act if act is not None else nn.Identity()
        self.rate = nn.Sequential(nn.Linear(in_dim, out_dim, bias=False),
                                  nn.Softplus(),
                                  nn.Dropout(drop))
        self.rob_bound = nn.Sequential(nn.Linear(in_dim, hid_dim, bias=True),
                                       nn.Softplus(),
                                       nn.Linear(hid_dim, out_dim, bias=True),
                                       nn.LayerNorm(out_dim),
                                       nn.Dropout(drop))
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x, edge_index, degree):
        """
        x : N * D
        edge_index: 2 * E
        degrees: N
        """
        rate = self.rate(x)
        gamma = self.rob_bound(x)
        x = self.fc(x)
        z = x
        x = self.drop(x)
        row, col = edge_index[0], edge_index[1]
        x = x[row] + x[col]
        x = scatter_sum(x, row, dim=0)
        x = (rate * x + gamma) / (1 + rate * degree.unsqueeze(1) + EPS) - z
        x = self.norm(x)
        return self.drop(self.act(x))


class BoundaryGCN(nn.Module):
    def __init__(self, n_layers, in_dim, hid_dim, embed_dim, out_dim, bias, act, drop=0.3):
        super().__init__()
        self.layers = nn.ModuleList([BoundaryConvLayer(in_dim, hid_dim, embed_dim, bias, act, drop)])
        self.res_lin = nn.ModuleList([nn.Linear(in_dim, embed_dim, bias=False) for _ in range(n_layers - 1)])
        for _ in range(n_layers - 2):
            self.layers.append(BoundaryConvLayer(embed_dim, hid_dim, embed_dim, bias, act, drop))
        self.layers.append(BoundaryConvLayer(embed_dim, hid_dim, out_dim, bias, None))

    def forward(self, data):
        x = data.x
        z = data.x
        edge_index = data.edge_index
        degree = data.degree
        for layer, res in zip(self.layers[:-1], self.res_lin):
            x = layer(x, edge_index, degree) + res(z)
        x = self.layers[-1](x, edge_index, degree)
        return x