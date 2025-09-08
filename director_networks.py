import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd
import tools
import numpy as np

class ActionHead(nn.Module):
    def __init__(
        self,
        inp_dim,
        size, # num_actions
        layers,
        units,
        act=nn.ELU,
        norm=nn.LayerNorm,
        dist="trunc_normal",
        init_std=0.0,
        min_std=0.1,
        max_std=1.0,
        temp=0.1,
        outscale=1.0,
        unimix_ratio=0.01,
    ):
        super(ActionHead, self).__init__()
        self._size = size
        self._layers = layers
        self._units = units
        self._dist = dist
        act = getattr(torch.nn, act)
        norm = getattr(torch.nn, norm)
        self._min_std = min_std
        self._max_std = max_std
        self._init_std = init_std
        self._unimix_ratio = unimix_ratio
        self._temp = temp() if callable(temp) else temp

        pre_layers = []
        for index in range(self._layers):
            pre_layers.append(nn.Linear(inp_dim, self._units, bias=False))
            pre_layers.append(norm(self._units, eps=1e-03))
            pre_layers.append(act())
            if index == 0:
                inp_dim = self._units
        self._pre_layers = nn.Sequential(*pre_layers)
        # self._pre_layers.apply(tools.weight_init)

        if isinstance(self._size, tuple):
            if self._dist != "onehot":
                print(f"only onehot is supported for 2d shape. got {self._dist}")
                raise NotImplementedError()
            self._dist_layer = nn.Linear(self._units, np.prod(self._size))
        else:
            if self._dist in ["tanh_normal", "tanh_normal_5", "normal", "trunc_normal"]:
                self._dist_layer = nn.Linear(self._units, 2 * self._size)
                # self._dist_layer.apply(tools.uniform_weight_init(outscale))
            elif self._dist in ["normal_1", "onehot", "onehot_gumbel"]:
                self._dist_layer = nn.Linear(self._units, self._size)
                # self._dist_layer.apply(tools.uniform_weight_init(outscale))

    def forward(self, features, dtype=None):
        x = features
        x = self._pre_layers(x)
        if self._dist == "tanh_normal":
            x = self._dist_layer(x)
            mean, std = torch.split(x, 2, -1)
            mean = torch.tanh(mean)
            std = F.softplus(std + self._init_std) + self._min_std
            dist = torchd.normal.Normal(mean, std)
            dist = torchd.transformed_distribution.TransformedDistribution(
                dist, tools.TanhBijector()
            )
            dist = torchd.independent.Independent(dist, 1)
            dist = tools.SampleDist(dist)
        elif self._dist == "tanh_normal_5":
            x = self._dist_layer(x)
            mean, std = torch.split(x, 2, -1)
            mean = 5 * torch.tanh(mean / 5)
            std = F.softplus(std + 5) + 5
            dist = torchd.normal.Normal(mean, std)
            dist = torchd.transformed_distribution.TransformedDistribution(
                dist, tools.TanhBijector()
            )
            dist = torchd.independent.Independent(dist, 1)
            dist = tools.SampleDist(dist)
        elif self._dist == "normal":
            x = self._dist_layer(x)
            mean, std = torch.split(x, [self._size] * 2, -1)
            std = (self._max_std - self._min_std) * torch.sigmoid(
                std + 2.0
            ) + self._min_std
            dist = torchd.normal.Normal(torch.tanh(mean), std)
            dist = tools.ContDist(torchd.independent.Independent(dist, 1))
        elif self._dist == "normal_1":
            x = self._dist_layer(x)
            dist = torchd.normal.Normal(mean, 1)
            dist = tools.ContDist(torchd.independent.Independent(dist, 1))
        elif self._dist == "trunc_normal":
            x = self._dist_layer(x)
            mean, std = torch.split(x, [self._size] * 2, -1)
            mean = torch.tanh(mean)
            std = 2 * torch.sigmoid(std / 2) + self._min_std
            dist = tools.SafeTruncatedNormal(mean, std, -1, 1)
            dist = tools.ContDist(torchd.independent.Independent(dist, 1))
        elif self._dist == "onehot":
            x = self._dist_layer(x)
            if type(self._size) == int:
                dist = tools.OneHotDist(x, unimix_ratio=self._unimix_ratio)
            elif len(self._size) == 2:
                x = x.reshape(x.shape[:-1] + self._size)
                dist = torchd.Independent(
                    tools.OneHotDist(x, unimix_ratio=self._unimix_ratio), 1
                )
                dist.minent = 0.0
                dist.maxent = np.prod(self._size[:-1]) * np.log(self._size[-1])
            else:
                raise NotImplementedError()
        elif self._dist == "onehot_gumble":
            x = self._dist_layer(x)
            temp = self._temp
            dist = tools.ContDist(torchd.gumbel.Gumbel(x, 1 / temp))
        else:
            raise NotImplementedError(self._dist)
        return dist