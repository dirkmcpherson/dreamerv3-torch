# blocks.py  -- block-sparse layers for Dreamer-style RSSM  (PyTorch ≥1.13)

import math, torch
from torch import nn, einsum


# -----------------------------------------------------------------------------#
# 1. BlockLinear: a fully-connected layer whose weight matrix is block-diagonal #
# -----------------------------------------------------------------------------#

class BlockLinear(nn.Module):
    """
    y = Wx + b, but W is block-diagonal with `g` blocks.
    Input  shape: (..., g * in_block)
    Output shape: (..., g * out_block)
    """
    def __init__(self, in_features: int, out_features: int, g: int = 8,
                 bias: bool = True):
        super().__init__()
        assert in_features % g == 0 and out_features % g == 0, \
            "in/out dim must be divisible by g"
        self.g = g
        self.in_block  = in_features // g
        self.out_block = out_features // g

        w_shape = (g, self.out_block, self.in_block)
        self.weight = nn.Parameter(torch.empty(w_shape))
        self.bias   = nn.Parameter(torch.empty(g, self.out_block)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_block)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        g, ib, ob = self.g, self.in_block, self.out_block
        # (..., g, ib)
        x = x.view(*x.shape[:-1], g, ib)
        # einsum → (..., g, ob)
        y = einsum('...gi,goi->...go', x, self.weight)
        if self.bias is not None:
            y = y + self.bias
        return y.flatten(-2, -1)      # (..., g*ob)


# -----------------------------------------------------------------------------#
# 2. BlockGRUCell: GRU whose gates use block-sparse affine transforms           #
# -----------------------------------------------------------------------------#

class BlockGRUCell(nn.Module):
    def __init__(self, inp_size: int, hidden_size: int,
                 g: int = 8, norm: bool = True,
                 act=torch.tanh, update_bias: float = -1.0):
        super().__init__()
        assert hidden_size % g == 0, "hidden size must be divisible by g"
        self._g = g
        self._hs = hidden_size
        self._act = act
        self._update_bias = update_bias

        self.lin = BlockLinear(inp_size + hidden_size, 3 * hidden_size, g,
                               bias=False)
        self.norm = nn.LayerNorm(3 * hidden_size, eps=1e-3) if norm else nn.Identity()

    @property
    def state_size(self):
        return self._hs

    def forward(self, x, state):        # state is list for compatibility
        h_prev = state[0]
        gates   = self.norm(self.lin(torch.cat([x, h_prev], -1)))
        reset, cand, update = torch.chunk(gates, 3, dim=-1)

        reset  = torch.sigmoid(reset)
        cand   = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        h_new  = update * cand + (1.0 - update) * h_prev
        return h_new, [h_new]
