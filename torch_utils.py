from torch import nn
import torch as th

class AutoAdapt(nn.Module):
  def __init__(
      self, shape, impl, scale, target, min, max,
      vel=0.1, thres=0.1, inverse=False):
    super(AutoAdapt, self).__init__()
    self._shape = shape
    self._impl = impl
    self._target = target
    self._min = min
    self._max = max
    self._vel = vel
    self._inverse = inverse
    self._thres = thres
    if self._impl == 'fixed':
      self._scale = th.tensor(scale)
    elif self._impl == 'mult':
      if len(shape) == 0:
        self._scale = 1.0
      else:
        self._scale = th.ones(shape, th.float32, requires_grad=False)
    elif self._impl == 'prop':
      self._scale = th.ones(shape, th.float32, requires_grad=False)
    else:
      raise NotImplementedError(self._impl)

  def __call__(self, reg, update=True):
    update and self.update(reg)
    scale = self.scale()
    loss = scale * (-reg if self._inverse else reg)
    metrics = {
        'mean': reg.mean(), 'std': reg.std(),
        'scale_mean': scale.mean(), 'scale_std': scale.std()}
    return loss, metrics

  def scale(self):
    if self._impl == 'fixed':
      scale = self._scale
    elif self._impl == 'mult':
      scale = self._scale
    elif self._impl == 'prop':
      scale = self._scale
    else:
      raise NotImplementedError(self._impl)
    return th.tensor(scale).detach()

  def update(self, reg):
    avg = reg.mean(list(range(len(reg.shape) - len(self._shape))))
    if self._impl == 'fixed':
      pass
    elif self._impl == 'mult':
      below = avg < (1 / (1 + self._thres)) * self._target
      above = avg > (1 + self._thres) * self._target
      if self._inverse:
        below, above = above, below
      inside = ~below & ~above
      # NOTE: How does the original implementation work?!
      adjusted = (
          above.float() * self._scale * (1 + self._vel) +
          below.float() * self._scale / (1 + self._vel) +
          inside.float() * self._scale)
      self._scale = th.clip(adjusted, self._min, self._max)
    elif self._impl == 'prop':
      direction = avg - self._target
      if self._inverse:
        direction = -direction
      self._scale = th.clip(self._scale + self._vel * direction, self._min, self._max)
    else:
      raise NotImplementedError(self._impl)