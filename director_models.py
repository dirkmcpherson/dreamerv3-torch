import copy
import torch
from torch import nn
import numpy as np
from PIL import ImageColor, Image, ImageDraw, ImageFont

import networks
import tools
import models as dv3_models

from IPython import embed as ipshell

to_np = lambda x: x.detach().cpu().numpy()

class HierarchyBehavior(nn.Module):
    def __init__(self, config, world_model, stop_grad_actor=True):
        super(HierarchyBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model
        self._stop_grad_actor = stop_grad_actor

        if config.dyn_discrete:
            z_dim = config.dyn_stoch * config.dyn_discrete # (z_stoch * z_discrete)
            feat_size = z_dim + config.dyn_deter # z_state + h
            gc_feat_size = z_dim + config.dyn_deter + z_dim # z_state + h + z_goal
            # feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter + # z_stoch * z_discrete + h + z_goal
        else:
            raise NotImplementedError

        
        # self.manager = networks.MLP(
        #     feat_size,
        #     (255,), # why?
        #     config.manager_layers,
        #     config.units,
        #     config.act,
        #     config.norm,
        #     dist=config.manager_dist,
        #     outscale=0.0,
        #     device=config.device,
        # )

        # the goal autencoder converts the feature representation (h? 1024?) to a one-hot of the goal
        self.goal_enc = networks.MLP(
            feat_size,
            z_dim, # size of output
            config.goal_ae_layers,
            config.goal_ae_units,
        )
        self.goal_dec = networks.MLP(
            z_dim,
            feat_size,
            config.goal_ae_layers,
            config.goal_ae_units,
        )

        self.manager = networks.ActionHead(
            feat_size,
            z_dim, # size out output
            config.manager_layers,
            config.manager_units,
            config.act,
            config.norm,
            config.manager_dist,
            config.manager_init_std,
            config.manager_min_std,
            config.manager_max_std,
            config.manager_temp,
            outscale=1.0,
            unimix_ratio=config.action_unimix_ratio,   
        )
        self.manager_exploration_value = networks.MLP(
            feat_size,  # pytorch version
            (255,),
            config.reward_layers,
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head,
            outscale=0.0,
            device=config.device,
        )
        self.manager_extrinsic_value = networks.MLP(
            feat_size,  # pytorch version
            (255,),
            config.reward_layers,
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head,
            outscale=0.0,
            device=config.device,
        )

        
        self.worker = networks.ActionHead( # worker is the worker policy
            gc_feat_size,  # pytorch version
            config.num_actions,
            config.actor_layers,
            config.units,
            config.act,
            config.norm,
            config.actor_dist,
            config.actor_init_std,
            config.actor_min_std,
            config.actor_max_std,
            config.actor_temp,
            outscale=1.0,
            unimix_ratio=config.action_unimix_ratio,
        )
        self.worker_value = networks.MLP(
            gc_feat_size,
            (255,),
            config.reward_layers,
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head,
            outscale=0.0,
            device=config.device,
        )

        if config.slow_value_target:
            self._slow_value_targets = {
                "worker": copy.deepcopy(self.worker_value), 
                "manager_exploration": copy.deepcopy(self.manager_exploration_value),
                "manager_extrinsic": copy.deepcopy(self.manager_extrinsic_value),
                }
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer(
            "worker",
            self.worker.parameters(),
            config.actor_lr,
            config.ac_opt_eps,
            config.actor_grad_clip,
            **kw,
        )
        self._worker_val_opt = tools.Optimizer(
            "worker_value",
            self.worker_value.parameters(),
            config.actor_lr,
            config.ac_opt_eps,
            config.actor_grad_clip,
            **kw,
        )

        self._manager_opt = tools.Optimizer(
            "manager",
            self.manager.parameters(),
            config.manager_lr,
            config.ac_opt_eps,
            config.manager_grad_clip,
            **kw,
        )
        self._manager_expl_opt = tools.Optimizer(
            "manager_exploration_value",
            self.manager_exploration_value.parameters(),
            config.manager_lr,
            config.ac_opt_eps,
            config.manager_grad_clip,
            **kw,
        )
        self._manager_extr_opt = tools.Optimizer(
            "manager_extrinsic_value",
            self.manager_extrinsic_value.parameters(),
            config.manager_lr,
            config.ac_opt_eps,
            config.manager_grad_clip,
            **kw,
        )

        if self._config.reward_EMA:
            self.reward_ema = dv3_models.RewardEMA(device=self._config.device)

    def _compute_target(
        self, imag_feat, imag_state, imag_action, reward, actor_ent, state_ent
    ):
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(reward)
        if self._config.future_entropy and self._config.actor_entropy() > 0:
            reward += self._config.actor_entropy() * actor_ent
        if self._config.future_entropy and self._config.actor_state_entropy() > 0:
            reward += self._config.actor_state_entropy() * state_ent
        value = self.value(imag_feat).mode()
        # value(15, 960, ch)
        # action(15, 960, ch)
        # discount(15, 960, ch)
        target = tools.lambda_return(
            reward[:-1],
            value[:-1],
            discount[:-1],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def _train(
        self,
        start,
        objective=None,
        action=None,
        reward=None,
        imagine=None,
        tape=None,
        repeats=None,
    ):
        print(f"Director skipping slow update")
        # self._update_slow_target()
        metrics = {}

        for k,v in start.items():
            print(f"{k}: {v.shape}")
        stoch, deter, logit = start["stoch"], start["deter"], start["logit"]

        with tools.RequiresGrad(self.manager):
            with torch.cuda.amp.autocast(self._use_amp):
                imag_feat, imag_state, imag_action, imag_goals = self._imagine(
                    start, self.worker, self._config.imag_horizon, repeats
                )
                reward = objective(imag_feat, imag_state, imag_action)

                imag_gc_feat = torch.cat([imag_feat, imag_goals], dim=-1)
                actor_ent = self.worker(imag_gc_feat).entropy()
                state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()
                # this target is not scaled
                # slow is flag to indicate whether slow_target is used for lambda-return
                target, weights, base = self._compute_target(
                    imag_feat, imag_state, imag_action, reward, actor_ent, state_ent
                )
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    imag_state,
                    imag_action,
                    target,
                    actor_ent,
                    state_ent,
                    weights,
                    base,
                )
                metrics.update(mets)


        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        if self._config.actor_dist in ["onehot"]:
            metrics.update(
                tools.tensorstats(
                    torch.argmax(imag_action, dim=-1).float(), "imag_action"
                )
            )
        else:
            metrics.update(tools.tensorstats(imag_action, "imag_action"))
        metrics["actor_ent"] = to_np(torch.mean(actor_ent))
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.worker.parameters()))
        return imag_feat, imag_state, imag_action, weights, metrics
    
    def _imagine(self, start, policy, horizon, repeats=None):
        dynamics = self._world_model.dynamics
        if repeats:
            raise NotImplemented("repeats is not implemented in this version")
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, step):
            state, _, _, goal = prev
            feat = dynamics.get_feat(state) # z_state + h
            inp = feat.detach() if self._stop_grad_actor else feat

            # NOTE: step is ignored in original code. does it mess with the flatten?
            if goal is None or step % self._config.goal_change_interval == 0:
                goal = self.manager(inp).sample()

            inp = torch.cat([inp, goal], -1)

            action = policy(inp).sample()

            # ipshell(); exit()
            succ = dynamics.img_step(state, action, sample=self._config.imag_sample)
            return succ, feat, action, goal

        succ, feats, actions, goals = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}
        if repeats:
            raise NotImplemented("repeats is not implemented in this version")

        return feats, states, actions, goals
    
    def _compute_actor_loss(
        self,
        imag_feat,
        imag_state,
        imag_action,
        target,
        actor_ent,
        state_ent,
        weights,
        base,
    ):
        metrics = {}
        inp = imag_feat.detach() if self._stop_grad_actor else imag_feat
        policy = self.actor(inp)
        actor_ent = policy.entropy()
        # Q-val for actor is not transformed using symlog
        target = torch.stack(target, dim=1)
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            values = self.reward_ema.values
            metrics["EMA_005"] = to_np(values[0])
            metrics["EMA_095"] = to_np(values[1])

        if self._config.imag_gradient == "dynamics":
            actor_target = adv
        elif self._config.imag_gradient == "reinforce":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
        elif self._config.imag_gradient == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = self._config.imag_gradient_mix()
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        if not self._config.future_entropy and (self._config.actor_entropy() > 0):
            actor_entropy = self._config.actor_entropy() * actor_ent[:-1][:, :, None]
            actor_target += actor_entropy
            metrics["actor_entropy"] = to_np(torch.mean(actor_entropy))
        if not self._config.future_entropy and (self._config.actor_state_entropy() > 0):
            state_entropy = self._config.actor_state_entropy() * state_ent[:-1]
            actor_target += state_entropy
            metrics["actor_state_entropy"] = to_np(torch.mean(state_entropy))
        actor_loss = -torch.mean(weights[:-1] * actor_target)
        return actor_loss, metrics

    def _update_slow_target(self):
        if self._config.slow_value_target:
            if self._updates % self._config.slow_target_update == 0:
                mix = self._config.slow_target_fraction
                for s, d in zip(self.worker_value.parameters(), self._slow_value_targets["worker"].parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
                for s, d in zip(self.manager_exploration_value.parameters(), self._slow_value_targets["manager_exploration"].parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
                for s, d in zip(self.manager_extrinsic_value.parameters(), self._slow_value_targets["manager_extrinsic"].parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1
