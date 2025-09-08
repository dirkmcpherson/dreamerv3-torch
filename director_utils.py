import models
import director_models as dmodels
import torch
from torch import nn
import torch as th
import tools

def get_model(example_env, config, dataset, logger):
    wm =  models.WorldModel(example_env.observation_space, example_env.action_space, step=None, config=config).to(config.device)

    z_size = config.dyn_stoch * config.dyn_discrete # one-hot vector grid size
    feat_size = z_size + config.dyn_deter

    goal_ae = dmodels.ImaginationAE(config).to(config.device)

    worker_critic_scales = {"reward_goal": 1.0} #{"reward": 0.5, "reward_goal": 0.5} 
    # worker_critic_scales = {"reward_goal": 1.0} if not WORKER_PRIMARY else {"reward": 1.0}
    worker_feat_size = feat_size + config.dyn_deter
    iac = worker = dmodels.ImaginationActorCritic(config, wm, input_size=worker_feat_size, num_actions=config.num_actions, value_heads=worker_critic_scales, stop_grad_actor=True, prefix="worker").to(config.device)
    print(f"Worker is using {worker.critics.keys()} critic(s)")

    # manager_critic_scales = {"reward": 0.9, "reward_expl": 0.1} # {"reward": 1.0} #
    # NOTE: if we're using human demonstrations the manager's entropy crashes early without getting any performance. Incentivize exploration more
    manager_critic_scales = {"reward": 0.6, "reward_expl": 0.4} # {"reward": 1.0} #
    manager = dmodels.ImaginationActorCritic(config, wm, input_size=feat_size, num_actions=config.skill_shape, value_heads=manager_critic_scales, stop_grad_actor=True, prefix="manager").to(config.device)
    print(f"manager is using {manager.critics.keys()} critic(s)")
    iagent = dmodels.ImaginationAgent(config, logger, worker=iac, manager=manager, goalae=goal_ae, _wm = wm, dataset=dataset)

    # if LOAD_PRETRAINED_WM:
    #     print("Loading pretrained wm")
    #     wm.load_state_dict(torch.load(logdir / "bk_wm_15000.pth")); wm.eval(); wm.requires_grad_(False)
    #     goal_ae.load_state_dict(torch.load(logdir / "bk_goalae_15000.pth")); goal_ae.eval(); goal_ae.requires_grad_(False)

    return iagent, worker, manager, goal_ae, wm


def imagine_with_skills(config, dynamics, start, policy, skill_policy, skill_decoder, horizon, starting_skill=None, starting_goal=None, sample=True):
    flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
    start = {k: flatten(v) for k, v in start.items()}

    if starting_skill is not None: # repeat the starting state to match the batch size. starting_skill is only done for human interaction
        start["deter"] = start["deter"].repeat(starting_goal.shape[1], 1)
        start["stoch"] = start["stoch"].repeat(starting_goal.shape[1], 1, 1)
        start["logit"] = start["logit"].repeat(starting_goal.shape[1], 1, 1)
        start = {k: v[None, ...] for k, v in start.items()}
        assert start["deter"].shape[:2] == starting_goal.shape[:2], f"{start['deter'].shape} {starting_goal.shape}"

    def step(prev, step):
        state, _, _, skill, goal = prev
        feat = dynamics.get_feat(state)

        if skill is None or (step % config.train_skill_duration == 0 and step > 0):
            skill = skill_policy(feat).sample()
            skill = skill.reshape([*skill.shape[:-2], -1])
            goal = skill_decoder(skill).mode()

        inp = feat.detach() if config.behavior_stop_grad else feat
        inp = torch.cat([inp, goal], -1)

        action = policy(inp).sample() if sample else policy(inp).mode()
        # succ = dynamics.img_step(state, action, sample=config.imag_sample)
        succ = dynamics.img_step(state, action, sample=sample)
        return succ, feat, action, skill, goal

    succ, feats, actions, skills, goals = tools.static_scan(
        step, [torch.arange(horizon)], (start, None, None, starting_skill, starting_goal)
    )
    states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}

    return feats, states, actions, skills, goals


from IPython import embed as ipshell

def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))

from collections.abc import Mapping
def deep_copy_dict(d, detach=False):
    # account for nested dicts
    if detach:
        return {k:deep_copy_dict(v, detach) if isinstance(v, Mapping) else v.detach().clone() for k,v in d.items()}
    else:
        return {k:deep_copy_dict(v) if isinstance(v, Mapping) else v.clone() for k,v in d.items()}
    
class AutoAdapt(nn.Module):
  def __init__(
      self, shape, impl, scale, target, min, max,
      vel=0.1, thres=0.1, inverse=False, device='cuda'):
    super(AutoAdapt, self).__init__()
    self._shape = shape
    self._impl = impl
    self._target = target
    self._min = min
    self._max = max
    self._vel = vel
    self._inverse = inverse
    self._thres = thres
    if self._impl == 'mult':
      if len(shape) == 0:
        self._scale = 1.0
      else:
        self._scale = th.ones(shape, dtype=th.float32, requires_grad=False).to(device)
    else:
      raise NotImplementedError(self._impl)

  def __call__(self, reg, minent=None, maxent=None, update=True):
    if minent is not None and maxent is not None:
      lo = minent / reg.shape[-1]; hi = maxent / reg.shape[-1]
      reg = (reg - lo) / (hi - lo)
    update and self.update(reg)
    scale = self.scale()
    loss = scale * (-reg if self._inverse else reg)
    metrics = {
        'mean': reg.mean(), 'std': reg.std(),
        'scale_mean': scale.mean(), 'scale_std': scale.std()}
    return loss, metrics

  def scale(self):
    scale = self._scale
    scale = self._scale
    scale = self._scale
    if type(scale) is th.Tensor:
      scale = scale.detach()
    else:
      scale = th.tensor(scale).detach()
    return scale

  def update(self, reg):
    avg = reg.mean(list(range(len(reg.shape) - len(self._shape))))
    if self._impl == 'mult':
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
      # if avg[0] < 0.55 and avg[0] > 0.45:
      # ipshell()
    else:
      raise NotImplementedError(self._impl)
    
class Ratio:
  def __init__(self, ratio):
    assert ratio >= 0, ratio
    self._ratio = ratio
    self._prev = None

  def __call__(self, step):
    step = int(step)
    if self._ratio == 0:
      return 0
    if self._prev is None:
      self._prev = step
      return 1
    repeats = int((step - self._prev) * self._ratio)
    self._prev += repeats / self._ratio
    return repeats