import copy
import torch
from torch import nn
import numpy as np
from PIL import ImageColor, Image, ImageDraw, ImageFont
import random

import networks
import tools
import models as dv3_models

from torchviz import make_dot
from IPython import embed as ipshell

to_np = lambda x: x.detach().cpu().numpy()

class HierarchyBehavior(nn.Module):
    def __init__(self, config, world_model, stop_grad_actor=True):
        super(HierarchyBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        if self._use_amp:
            raise NotImplementedError("AMP is not implemented in this version")
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

        

        # NOTE: The RSSM has MLPs that support a nxn goal space. Use that to figure out why this isn't working. 

        # the goal autencoder converts the feature representation (h? 1024?) to a one-hot of the goal
        self.goal_enc = networks.MLP(
            # feat_size,
            config.dyn_deter,
            # z_dim,
            config.skill_shape, # size of output
            **config.goal_encoder
        )
        self.goal_dec = networks.MLP(
            np.prod(config.skill_shape),
            # (config.dyn_stoch, config.dyn_discrete), # size of input
            # feat_size,
            config.dyn_deter,
            **config.goal_decoder
        )
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self.goal_ae_opt = tools.Optimizer(
            "goal_ae",
            list(self.goal_enc.parameters()) + list(self.goal_dec.parameters()),
            config.model_lr,
            config.ac_opt_eps,
            config.goal_ae_grad_clip,
            **kw,
        )

        z_shape = config.dyn_stoch * config.dyn_discrete if config.dyn_discrete else config.dyn_stoch
        feat_size = config.dyn_deter + z_shape
        '''
        Manager takes the world state {h,z} and outputs a skill (z_goal)
        '''
        self.manager = ImagActorCritic(config, world_model, input_shape=feat_size, num_actions=config.skill_shape, stop_grad_actor=stop_grad_actor, prefix="manager")
        
        '''
        Worker takes the world state and skill {h,z,z_goal} and outputs an action
        '''
        self.worker = ImagActorCritic(config, world_model, input_shape=feat_size+z_shape, num_actions=config.num_actions, stop_grad_actor=stop_grad_actor, prefix="worker")

        # NOTE: Hafner has a custom one-hot but it looks like straightthrough
        # self.skill_prior = tools.OneHotDist(logits=torch.zeros(config.dyn_stoch, config.dyn_discrete))
        self.skill_prior = torch.distributions.independent.Independent(
                tools.OneHotDist(logits=torch.zeros(*config.skill_shape, device=config.device), unimix_ratio=0.0), 1
            )
    
    def goal_pred(self, data):
        data = self._world_model.preprocess(data)
        reshape = lambda x: x.reshape([*x.shape[:-2], -1])

        # Look at a random subset of batches 
        random_batches = random.choices(range(data["image"].shape[0]), k=4)
        random_samples = random.choices(range(data["image"].shape[1]), k=1)

        actions = data["action"][random_batches, :]
        is_first = data["is_first"][random_batches, :]
        embed = self._world_model.encoder(data)[random_batches, :]

        states, _ = self._world_model.dynamics.observe(embed, actions, is_first)

        # feat = self._world_model.dynamics.get_feat(states)
        feat = states["deter"]
        enc = self.goal_enc(feat).mode()
        enc = reshape(enc) # (time, batch, feat0*feat1)
        dec = self.goal_dec(enc).mode()
        deter = dec
        
        # now add the stochastic state to the deterministic state, which is what the world model decoder requires
        stoch = self._world_model.dynamics.get_stoch(deter)
        stoch = reshape(stoch)
        inp = torch.cat([stoch, deter], dim=-1)

        model = self._world_model.heads["decoder"](inp)["image"].mode() + 0.5
        truth = data["image"][random_batches, :] + 0.5
        error = (model - truth + 1.0) / 2.0

        model = model[:, random_samples]
        truth = truth[:, random_samples]
        error = error[:, random_samples]
        
        # return model
        # print(f"director_models::goal_pred"); ipshell()
        return truth, model, error


    def _compute_manager_target(
        self, imag_feat, imag_state, imag_action, extr_reward, actor_ent, state_ent
    ):
        '''
        Compute the manager extrinsic target, which is the lamda-return over the horizon
        '''
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(extr_reward)
        if self._config.future_entropy and self._config.actor_entropy() > 0:
            extr_reward += self._config.actor_entropy() * actor_ent
        if self._config.future_entropy and self._config.actor_state_entropy() > 0:
            extr_reward += self._config.actor_state_entropy() * state_ent
        value = self.manager_extrinsic_value(imag_feat).mode()
        # value(15, 960, ch)
        # action(15, 960, ch)
        # discount(15, 960, ch)
        target = tools.lambda_return(
            extr_reward[:-1],
            value[:-1],
            discount[:-1],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        # print(f"director_models::_compute_manager_target"); ipshell()
        return target, weights, value[:-1]
    
    def extr_reward(self, state):
        # extrR = self._world_model.heads["reward"](self._world_model.dynamics.get_feat(state)).mode() # NOTE: Original uses mean()[1:] 
        extrR = self._world_model.heads["reward"](self._world_model.dynamics.get_feat(state)).mean()[1:]
        return extrR

    def expl_reward(self, feat, state, action):
        # elbo (Evidence Lower BOund) reward
        # NOTE: hafner's code uses a context variable whose use I don't understand.
        feat = state["deter"]
        enc = self.goal_enc(feat)
        x = enc.sample()
        x = x.reshape([x.shape[0], x.shape[1], -1]) # (time, batch, feat0*feat1)
        dec = self.goal_dec(x)

        # ll = dec.log_prob(feat)
        # kl = torch.distributions.kl.kl_divergence(enc, self.skill_prior)

        return ((dec.mode() - feat) ** 2).mean(-1)[1:] # NOTE: This can probably be a log likelihood under the MSE distribution
        # context = tf.repeat(feat[0][None], 1 + self.config.imag_horizon, 0)
        # enc = self.enc({'goal': feat, 'context': context})
        # dec = self.dec({'skill': enc.sample(), 'context': context})
        # ll = dec.log_prob(feat)
        # kl = tfd.kl_divergence(enc, self.prior)
        # return ((dec.mode() - feat) ** 2).mean(-1)[1:]

    def goal_reward(self, feat, state, action, goal):
        # cosine_max reward
        h = state["deter"]
        goal = goal.detach()
        gnorm = torch.linalg.norm(goal, dim=-1, keepdim=True) + 1e-12
        hnorm = torch.linalg.norm(h, dim=-1, keepdim=True) + 1e-12
        norm = torch.max(gnorm, hnorm)
        return torch.einsum("...i,...i->...", goal / norm, h / norm)[1:] # NOTE

    def split_traj(self, traj):
        traj = traj.copy()
        k = self._config.train_skill_duration
        # print(f"Trajectory length must be divisible by k+1 {len(traj['action'])} % {k} != 1")
        assert len(traj['action']) % k == 1; (len(traj['action']) % k), "Trajectory length must be divisible by k+1"
        reshape = lambda x: x.reshape([x.shape[0] // k, k] + list(x.shape[1:]))
        for key, val in list(traj.items()):
            # print(f"director_models::split_traj::key: {key}, val.shape: {val.shape if hasattr(val, 'shape') else None}")
            val = torch.concat([0 * val[:1], val], 0) if 'reward' in key else val
            # (1 2 3 4 5 6 7 8 9 10) -> ((1 2 3 4) (4 5 6 7) (7 8 9 10))
            val = torch.concat([reshape(val[:-1]), val[k::k][:, None]], 1)
            # N val K val B val F... -> K val (N B) val F...
            val = val.permute([1, 0] + list(range(2, len(val.shape))))
            val = val.reshape(
                [val.shape[0], np.prod(val.shape[1:3])] + list(val.shape[3:]))
            val = val[1:] if 'reward' in key else val
            traj[key] = val
        # Bootstrap sub trajectory against current not next goal.
        traj['goal'] = torch.concat([traj['goal'][:-1], traj['goal'][:1]], 0)
        traj['weight'] = torch.cumprod(
            self._config.discount * traj['cont'], axis=0) / self._config.discount
        return traj

    def abstract_traj(self, orig_traj):
        traj = orig_traj.copy()
        traj["action"] = traj.pop("goal")
        k = self._config.train_skill_duration
        reshape = lambda x: x.reshape([x.shape[0] // k, k] + list(x.shape[1:]))
        weights = torch.cumprod(reshape(traj["cont"][:-1]), 1).to(self._config.device)
        for key, value in list(traj.items()):
            if 'reward' in key:
                traj[key] = (reshape(value) * weights).mean(1)
            elif key == 'cont':
                traj[key] = torch.concat([value[:1], reshape(value[1:]).prod(1)], 0)
            else:
                traj[key] = torch.concat([reshape(value[:-1])[:, 0], value[-1:]], 0)
        traj['weight'] = torch.cumprod(
            self._config.discount * traj['cont'], 0) / self._config.discount
    
        return traj

    def train_goal_vae(self, start, metrics):
        # context feat is thes 
        # feat = context["feat"].detach()
        feat = start["deter"].detach()
        feat.requires_grad = True

        enc = self.goal_enc(feat)
        x = enc.sample()
        x = x.reshape([x.shape[0], x.shape[1], -1]) # (time, batch, feat0*feat1)
        dec = self.goal_dec(x)
        
        # NOTE: Should kl and recreation loss be scaled by scheduled constants as in the world model?
        rec = -dec.log_prob(feat.detach())
        kl = torch.distributions.kl.kl_divergence(enc, self.skill_prior)
        loss = torch.mean(rec + kl)
        
        # dot = make_dot(loss, params=dict(list(self.goal_enc.named_parameters()) + list(self.goal_dec.named_parameters())), show_attrs=True, show_saved=True)
        # dot.format = "png"
        # dot.render("goal_ae_loss", directory="./graphs")

        # with tools.RequiresGrad(self):
        metrics.update(tools.tensorstats(rec, "goal_ae_rec"))
        metrics.update(tools.tensorstats(kl, "goal_ae_kl"))
        metrics.update(tools.tensorstats(loss, "goal_ae_loss"))

        self.goal_ae_opt(loss, list(self.goal_enc.parameters()) + list(self.goal_dec.parameters()))
        # print(f"director_models::_train::train_goal_vae"); ipshell()
        return metrics
    
    def _train(
        self,
        start,
        context=None,
        action=None,
        extr_reward=None,
        imagine=None,
        tape=None,
        repeats=None,
    ):
        metrics = {}

        # Train the goal autoencoder on world model representations
        # NOTE: These are posterior representations of the world model (s_t|s_t-1, a_t-1, x_t)
        metrics = self.train_goal_vae(start, metrics)

        if self._config.debug_only_goal_ae:
            return [metrics]

        ## Manager updates
        # with tools.RequiresGrad(self.manager): # NOTE: The worker needs a grad too, how do these two interact?
        #     with torch.cuda.amp.autocast(self._use_amp):
        # Given the output world model starting state, do imagined rollouts at each step with the worker's action policy 
        imag_feat, imag_state, imag_action, imag_goals = self._imagine_carry(
            start, self.worker.actor, self._config.imag_horizon, repeats
        )

        # compute the rewards 
        reward_extr = self.extr_reward(imag_state)
        reward_expl = self.expl_reward(imag_feat, imag_state, imag_action)
        reward_goal = self.goal_reward(imag_feat, imag_state, imag_action, imag_goals)

        reward_expl = reward_expl.unsqueeze(-1)
        reward_goal = reward_goal.unsqueeze(-1)


        # The rollout needs to be split two ways. 
        # The manager takes every kth step and sums the weighted rewards for 0 to k-1
        # The worker is trained on k step snippets that have the same goal
        # ipshell()
        traj = {
            "stoch": imag_state["stoch"],
            "deter": imag_state["deter"],
            "logit": imag_state["logit"],
            "feat": imag_feat,
            "action": imag_action,
            "reward_extr": reward_extr,
            "reward_expl": reward_expl,
            "reward_goal": reward_goal,
            "goal": imag_goals,
            "cont": torch.zeros(list(imag_action.shape[:-1]) + [1]) #TODO: Need to get continuation binary values. Are these from raw data? or is there an imaginary continuation?
        }

        ipshell()

        for k,v in traj.items():
            print(f"{k}: {v.shape if hasattr(v, 'shape') else 'None'}")
            # batch, time, feature -> time, batch, feature
            traj[k] = v.permute(1,0)

        # worker trajecory must be split into goal horizon length chunks
        wtraj = self.split_traj(traj)
        wtraj["reward"] = wtraj["reward_goal"]
        wtraj["state"] = {"stoch": wtraj["stoch"], "deter": wtraj["deter"], "logit": wtraj["logit"]}

        # manager trajectory must be split to only include the goal selection steps and sum reward between them
        mtraj = self.abstract_traj(traj)
        mtraj["reward"] = self._config.expl_reward_weight * mtraj["reward_expl"] + self._config.extr_reward_weight * mtraj["reward_extr"] # TODO: Weight rewards
        mtraj["state"] = {"stoch": mtraj["stoch"], "deter": mtraj["deter"], "logit": mtraj["logit"]}

        print(f"original action dim: {imag_action.shape} --> {wtraj['action'].shape}")
        print(f"                     {imag_action.shape} --> {mtraj['action'].shape}")

        print(f"Manager Traj")
        for key, value in list(mtraj.items()):
            if key == "action":
                old = traj["goal"]
            elif key not in traj:
                print(f"\t{key} shape:  None --> {value.shape if hasattr(value, 'shape') else 'None'}")
                continue
            else:
                old = traj[key]
            print(f"\t{key} shape:  {old.shape} --> {value.shape}")
            
        print(f"Worker Traj")
        for key, value in list(wtraj.items()):
            if key not in traj:
                print(f"\t{key} shape:  None --> {value.shape if hasattr(value, 'shape') else 'None'}")
                continue
            else:
                old = traj[key]
            print(f"\t{key} shape:  {old.shape} --> {value.shape}")

        metrics.update(self.manager._train(imag_traj=mtraj)[-1])
        metrics.update(self.worker._train(imag_traj=wtraj)[-1])

        return None, metrics
    
    def _imagine(self, start_wm, policy, horizon, repeats=None):
        dynamics = self._world_model.dynamics
        if repeats:
            raise NotImplemented("repeats is not implemented in this version")
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start_wm.items()}

        def step(prev, step):
            state, _, _, goal = prev
            feat = dynamics.get_feat(state) # z_state + h
            inp = feat.detach() if self._stop_grad_actor else feat

            # NOTE: step is ignored in original code. does it mess with the flatten?
            if goal is None or step % self._config.train_skill_duration == 0:
                goal = self.manager(inp).sample()

            inp = torch.cat([inp, goal], -1)

            action = policy(inp).sample()

            # print(f"director_models::_imagine::step"); ipshell()
            succ = dynamics.img_step(state, action, sample=self._config.imag_sample)
            return succ, feat, action, goal

        succ, feats, actions, goals = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None, None)
        )
        states = {k: torch.cat([start[k].unsqueeze(0), v[:-1]], 0) for k, v in succ.items()}
        if repeats:
            raise NotImplemented("repeats is not implemented in this version")

        # print(f"director_models::_imagine"); ipshell()
        return feats, states, actions, goals

    def _imagine_carry(self, start_wm, policy, horizon, repeats=None):
        dynamics = self._world_model.dynamics
        if repeats:
            raise NotImplemented("repeats is not implemented in this version")
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start_wm.items()}

        feat = dynamics.get_feat(start)
        inp = feat.detach() if self._stop_grad_actor else feat
        goal = self.sample_goal(feat)

        inp = torch.cat([inp, goal], -1)
        action = policy(inp).sample()

        actions = [action]
        states = start
        feats = [feat]
        goals = [goal]

        def step(prev, step):
            state, _, _, goal = prev
            feat = dynamics.get_feat(state) # z_state + h
            inp = feat.detach() if self._stop_grad_actor else feat

            # NOTE: step is ignored in original code. does it mess with the flatten?
            if goal is None or step % self._config.train_skill_duration == 0:
                goal = self.sample_goal(feat)

            inp = torch.cat([inp, goal], -1)

            action = policy(inp).sample()

            # print(f"director_models::_imagine::step"); ipshell()
            succ = dynamics.img_step(state, action, sample=self._config.imag_sample)
            return succ, feat, action, goal

        succ, sc_feats, sc_actions, sc_goals = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None, goal)
        )
        sc_states = {k: torch.cat([start[k].unsqueeze(0), v[:-1]], 0) for k, v in succ.items()}
        if repeats:
            raise NotImplemented("repeats is not implemented in this version")

        feats = torch.cat([torch.stack(feats), sc_feats], dim=0)
        actions = torch.cat([torch.stack(actions), sc_actions], dim=0)
        goals = torch.cat([torch.stack(goals), sc_goals], dim=0)
        for k,v in sc_states.items():
            states[k] = torch.cat([states[k].unsqueeze(0), v], dim=0)

        # for idx, (f,s,a,g) in enumerate(zip(feats, states, actions, goals)):
        #     print(f"feat: {f.shape}, state: {s.shape}, action: {a.shape}, goal: {g.shape} IDX: {idx}")

        # print(f"director_models::_imagine"); ipshell()
        return feats, states, actions, goals

    
    def _compute_manager_loss(
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
        policy = self.manager(inp)
        # policy = self.actor(inp)
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
                * (target - self.manager_extrinsic_value(imag_feat[:-1]).mode()).detach()
            )
        elif self._config.imag_gradient == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.manager_extrinsic_value(imag_feat[:-1]).mode()).detach()
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
    
    def sample_goal(self, feat):
        goal = self.manager.actor(feat).sample().reshape(-1, np.product(self._config.skill_shape))
        return self.goal_dec(goal).mode()

class ImagActorCritic(nn.Module):
    def __init__(self, config, world_model, input_shape, num_actions, stop_grad_actor=True, prefix=''):
        super(ImagActorCritic, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model
        self._stop_grad_actor = stop_grad_actor
        self._prefix = prefix

        if prefix == "manager":
            actor_kw = config.manager_actor
        elif prefix == "worker":
            actor_kw = config.worker_actor
        else:
            raise NotImplementedError(prefix)

        self.actor = networks.ActionHead(
            input_shape,
            num_actions,
            config.actor_layers,
            config.units,
            actor_kw['act'],
            actor_kw['norm'],
            actor_kw['dist'],
            config.actor_init_std,
            config.actor_min_std,
            config.actor_max_std,
            config.actor_temp,
            outscale=actor_kw['outscale'],
            unimix_ratio=config.action_unimix_ratio,
        )
        
        # if config.value_head == "symlog_disc": # NOTE: assumed to be true
        self.value = networks.MLP(
            input_shape,
            (255,),
            config.value_layers,
            config.units,
            config.act,
            config.norm,
            config.value_head,
            outscale=0.0,
            device=config.device,
        )

        self.expl_value = networks.MLP(
            input_shape,
            (255,),
            config.value_layers,
            config.units,
            config.act,
            config.norm,
            config.value_head,
            outscale=0.0,
            device=config.device,
        ) if self._prefix == "manager" else None
        
        if config.slow_value_target:
            self._slow_value = copy.deepcopy(self.value)
            self._slow_value_expl = copy.deepcopy(self.expl_value) if self.expl_value is not None else None
            self._updates = 0
            
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer(
            f"{self._prefix}_actor",
            self.actor.parameters(),
            config.actor_lr,
            config.ac_opt_eps,
            config.actor_grad_clip,
            **kw,
        )
        self._value_opt = tools.Optimizer(
            f"{self._prefix}_value",
            self.value.parameters(),
            config.value_lr,
            config.ac_opt_eps,
            config.value_grad_clip,
            **kw,
        )
        self._expl_value_opt = tools.Optimizer(
            f"{self._prefix}_expl_value",
            self.expl_value.parameters(),
            config.value_lr,
            config.ac_opt_eps,
            config.value_grad_clip,
            **kw,
        ) if self.expl_value is not None else None
        if self._config.reward_EMA:
            self.reward_ema = dv3_models.RewardEMA(device=self._config.device)

    def _train(
        self,
        start=None,
        objective=None,
        action=None,
        reward=None,
        imagine=None,
        tape=None,
        repeats=None,
        imag_traj=None,
    ):
        if start == None and imag_traj == None:
            raise ValueError("Must provide either start or imag_traj")
        
        self._update_slow_target()
        metrics = {}

        with tools.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp):
                if imag_traj is None:
                    imag_feat, imag_state, imag_action = self._imagine(
                        start, self.actor, self._config.imag_horizon, repeats
                    )
                    rewards = objective(imag_feat, imag_state, imag_action)
                    # reduce the rewards to a single entry for each timestep
                    imag_reward = sum(rewards)
                else:
                    imag_feat = imag_traj["feat"]
                    imag_state = imag_traj["state"]
                    imag_action = imag_traj["action"]
                    imag_reward = imag_traj["reward"]
                    imag_goal = imag_traj["goal"] if "goal" in imag_traj else None # NOTE: goals become actions for the manager

                if self._prefix == "manager":
                    inp = imag_feat
                elif self._prefix == "worker":
                    inp = torch.cat([imag_feat, imag_goal], dim=-1)
                else:
                    raise NotImplementedError(self._prefix)

                actor_ent = self.actor(inp).entropy()
                state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()
                # this target is not scaled
                # slow is flag to indicate whether slow_target is used for lambda-return
                target, weights, base = self._compute_target(
                    imag_feat, imag_state, imag_action, imag_goal, imag_reward, actor_ent, state_ent
                )
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    imag_state,
                    imag_action,
                    imag_goal,
                    target,
                    actor_ent,
                    state_ent,
                    weights,
                    base,
                )
                metrics.update(mets)
                value_input = imag_feat

        with tools.RequiresGrad(self.value):
            with torch.cuda.amp.autocast(self._use_amp):
                # value = self.value(value_input[:-1].detach())
                value = self.value(value_input[:-1].detach())
                target = torch.stack(target, dim=1)
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = -value.log_prob(target.detach())
                slow_target = self._slow_value(value_input[:-1].detach())
                if self._config.slow_value_target:
                    value_loss = value_loss - value.log_prob(
                        slow_target.mode().detach()
                    )
                if self._config.value_decay:
                    value_loss += self._config.value_decay * value.mode()
                # (time, batch, 1), (time, batch, 1) -> (1,)
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        if self.expl_value is not None:
            with tools.RequiresGrad(self.expl_value):
                with torch.cuda.amp.autocast(self._use_amp):
                    # value = self.value(value_input[:-1].detach())
                    expl_value = self.expl_value(value_input[:-1].detach())
                    target = torch.stack(target, dim=1)
                    # (time, batch, 1), (time, batch, 1) -> (time, batch)
                    expl_value_loss = -expl_value.log_prob(target.detach())
                    slow_target = self._slow_value_expl(value_input[:-1].detach())
                    if self._config.slow_value_target:
                        expl_value_loss = expl_value_loss - expl_value.log_prob(
                            slow_target.mode().detach()
                        )
                    if self._config.value_decay:
                        expl_value_loss += self._config.value_decay * expl_value.mode()
                    # (time, batch, 1), (time, batch, 1) -> (1,)
                    expl_value_loss = torch.mean(weights[:-1] * expl_value_loss[:, :, None])

        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        if self.expl_value is not None:
            metrics.update(tools.tensorstats(expl_value.mode(), "expl_value"))
            metrics.update(tools.tensorstats(expl_value_loss, "expl_value_loss"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        if self._config.actor_dist in ["onehot"]:
            metrics.update(
                tools.tensorstats(
                    torch.argmax(imag_action, dim=-1).float(), "imag_action"
                )
            )
        else:
            metrics.update(tools.tensorstats(imag_action, "imag_action"))
        metrics["actor_entropy"] = to_np(torch.mean(actor_ent))
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
            if self.expl_value is not None:
                metrics.update(self._expl_value_opt(expl_value_loss, self.expl_value.parameters()))
        return imag_feat, imag_state, imag_action, weights, metrics

    def _imagine(self, start, policy, horizon, repeats=None):
        dynamics = self._world_model.dynamics
        if repeats:
            raise NotImplemented("repeats is not implemented in this version")
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach() if self._stop_grad_actor else feat
            action = policy(inp).sample()
            succ = dynamics.img_step(state, action, sample=self._config.imag_sample)
            return succ, feat, action

        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}
        if repeats:
            raise NotImplemented("repeats is not implemented in this version")

        return feats, states, actions

    def _compute_target(
        self, imag_feat, imag_state, imag_action, imag_goal, reward, actor_ent, state_ent
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

        if self._prefix == "manager":
            inp = imag_feat
        elif self._prefix == "worker":
            inp = torch.cat([imag_feat, imag_goal], dim=-1)
        else:
            raise NotImplementedError(self._prefix)
        
        value = self.calc_value(inp)
        # value = self.value(inp).mode()
        # if self.expl_value is not None:
        #     value = value + self.expl_value(inp).mode()
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

    def calc_value(self, inp):
        value = self._config.extr_reward_weight*self.value(inp).mode()
        if self.expl_value is not None:
            value = value + self._config.expl_reward_weight*self.expl_value(inp).mode()
        return value

    def _compute_actor_loss(
        self,
        imag_feat,
        imag_state,
        imag_action,
        imag_goal,
        target,
        actor_ent,
        state_ent,
        weights,
        base,
    ):
        metrics = {}
        if self._prefix == "manager":
            inp = imag_feat
        elif self._prefix == "worker":
            inp = torch.cat([imag_feat, imag_goal], dim=-1)
        else:
            raise NotImplementedError(self._prefix)
        inp = inp.detach() if self._stop_grad_actor else inp
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
                # * (target - self.value(imag_feat[:-1]).mode()).detach()
                * (target - self.calc_value(imag_feat[:-1])).detach()
            )
        elif self._config.imag_gradient == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                # * (target - self.value(imag_feat[:-1]).mode()).detach()
                * (target - self.calc_value(imag_feat[:-1])).detach()
            )
            mix = self._config.imag_gradient_mix()
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        if not self._config.future_entropy and (self._config.actor_entropy() > 0):
            actor_entropy = self._config.actor_entropy() * actor_ent[:-1][:, :, None]
            actor_target += actor_entropy
        if not self._config.future_entropy and (self._config.actor_state_entropy() > 0):
            state_entropy = self._config.actor_state_entropy() * state_ent[:-1]
            actor_target += state_entropy
            metrics["actor_state_entropy"] = to_np(torch.mean(state_entropy))
        actor_loss = -torch.mean(weights[:-1] * actor_target)

        prefixxed_metrics = {}
        for k,v in metrics.items():
            prefixxed_metrics[self._prefix+"_"+k] = v
        return actor_loss, prefixxed_metrics

    def _update_slow_target(self):
        if self._config.slow_value_target:
            if self._updates % self._config.slow_target_update == 0:
                mix = self._config.slow_target_fraction
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
                if self.expl_value is not None:
                    for s, d in zip(self.expl_value.parameters(), self._slow_value_expl.parameters()):
                        d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1
