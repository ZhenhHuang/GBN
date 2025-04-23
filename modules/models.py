import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum


EPS = 1e-4


class BoundaryConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias, act=None):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=bias)
        self.act = act if act is not None else nn.Identity()
        self.rate = nn.Linear(in_dim, out_dim, bias=False)
        self.rob_bound = nn.Linear(in_dim, out_dim, bias=True)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index, degree):
        """
        x : N * D
        edge_index: 2 * E
        degrees: N
        """
        rate = F.softplus(self.rate(x)) + EPS
        gamma = self.rob_bound(x)
        x = self.fc(x)
        row, col = edge_index[0], edge_index[1]
        x = x[row] + x[col]
        x = scatter_sum(x, row, dim=0)
        x = (rate * x + gamma) / (1 + rate * degree.unsqueeze(1) + EPS)
        x = self.norm(x)
        return self.act(x)


class BoundaryGCN(nn.Module):
    def __init__(self, n_layers, in_dim, embed_dim, out_dim, bias, act):
        super().__init__()
        self.layers = nn.ModuleList([BoundaryConvLayer(in_dim, embed_dim, bias, act)])
        self.res_lin = nn.ModuleList([nn.Linear(in_dim, embed_dim, bias=False) for _ in range(n_layers - 1)])
        for _ in range(n_layers - 2):
            self.layers.append(BoundaryConvLayer(embed_dim, embed_dim, bias, act))
        self.layers.append(BoundaryConvLayer(embed_dim, out_dim, bias, None))

    def forward(self, data):
        x = data.x
        z = data.x
        edge_index = data.edge_index
        degree = data.degree
        for layer, res in zip(self.layers[:-1], self.res_lin):
            x = layer(x, edge_index, degree) + res(z)
        x = self.layers[-1](x, edge_index, degree)
        return x

