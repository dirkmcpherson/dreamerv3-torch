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
            for k,v in self._slow_value_targets.items():
                v.to(config.device)

            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self.goal_ae_opt = tools.Optimizer(
            "goal_ae",
            list(self.goal_enc.parameters()) + list(self.goal_dec.parameters()),
            config.model_lr,
            config.ac_opt_eps,
            config.goal_ae_grad_clip,
            **kw,
        )
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
        random_samples = random.choices(range(data["image"].shape[1]), k=2)

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
        extrR = self._world_model.heads["reward"](self._world_model.dynamics.get_feat(state)).mode() # NOTE: Original uses mean()[1:] 
        return extrR

    def expl_reward(self, feat, state, action):
        # elbo (Evidence Lower BOund) reward
        # NOTE: hafner's code uses a context variable whose use I don't understand.
        feat = self._world_model.dynamics.get_feat(state)
        enc = self.goal_enc(feat)
        x = enc.sample()
        x = x.reshape([x.shape[0], x.shape[1], -1]) # (time, batch, feat0*feat1)
        dec = self.goal_dec(x)

        # ll = dec.log_prob(feat)
        # kl = torch.distributions.kl.kl_divergence(enc, self.skill_prior)

        return ((dec.mode() - feat) ** 2).mean(-1)[1:]
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
        traj.copy()
        k = self._config.train_skill_duration
        # print(f"Trajectory length must be divisible by k+1 {len(traj['action'])} % {k} != 1")
        assert len(traj['action']) % k == 1; (len(traj['action']) % k), "Trajectory length must be divisible by k+1"
        reshape = lambda x: x.reshape([x.shape[0] // k, k] + list(x.shape[1:]))
        for key, val in list(traj.items()):
            if "reward" in key:
                ipshell()
            print(f"director_models::split_traj::key: {key}, val.shape: {val.shape}")
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
        traj['weight'] = torch.math.cumprod(
            self.config.discount * traj['cont']) / self.config.discount
        return traj
        pass

    def abstract_traj(self, traj):
        traj.copy()
        k = self._config.train_skill_duration
        pass

    def _train(
        self,
        start,
        context,
        action=None,
        extr_reward=None,
        imagine=None,
        tape=None,
        repeats=None,
    ):
        self._update_slow_target()
        metrics = {}

        # for k,v in start.items():
        #     print(f"{k}: {v.shape}")
        # stoch, deter, logit = start["stoch"], start["deter"], start["logit"]
        # print(f"director_models::_train"); ipshell()

        # NOTE: context["feat"] == rssm::get_feat(state)
        # Assert this is true
        # assert torch.all(torch.eq(context["feat"], self._world_model.dynamics.get_feat(start)))
        # print("context feat == start feat. NOTE: Remove this for real training.")

        # make_dot(context["feat"], show_attrs=True, show_saved=True).save("feat.dot", directory="./graphs")
        def train_goal_vae(start, context, metrics):
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

        # Train the goal autoencoder on world model representations
        # NOTE: These are posterior representations of the world model (s_t|s_t-1, a_t-1, x_t)
        metrics = train_goal_vae(start, context, metrics)

        if self._config.debug_only_goal_ae:
            return [metrics]

        ## Manager updates
        # with tools.RequiresGrad(self.manager):
        #     with torch.cuda.amp.autocast(self._use_amp):

        # Given the output world model starting state, do imagined rollouts at each step with the worker's action policy 
        imag_feat, imag_state, imag_action, imag_goals = self._imagine_carry(
            start, self.worker, self._config.imag_horizon, repeats
        )

        reward_extr = self.extr_reward(imag_state)
        reward_expl = self.expl_reward(imag_feat, imag_state, imag_action)
        reward_goal = self.goal_reward(imag_feat, imag_state, imag_action, imag_goals)

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

        wtraj = self.split_traj(traj)
        mtraj = self.abstract_traj(traj)


        ## Goal autoencoder
        # conver the horizon, batchsteps, feat0, feat1 to horizon, batchsteps, feat0*feat1
        # imag_stoch = imag_state["stoch"].reshape(imag_state["stoch"].shape[0], imag_state["stoch"].shape[1], -1)

        imag_gc_feat = torch.cat([imag_feat, imag_goals], dim=-1)
        actor_ent = self.worker(imag_gc_feat).entropy()
        mgr_ent = self.manager(imag_feat).entropy()
        state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()
        # this target is not scaled
        # slow is flag to indicate whether slow_target is used for lambda-return
        manager_extrinsic_target, weights, base = self._compute_manager_target(
            imag_feat, imag_state, imag_action, reward_extr, mgr_ent, state_ent
        )

        # ipshell()
        manager_exploration_target, weights, base = self._compute_manager_target(
            imag_feat, imag_state, imag_action, reward_expl, mgr_ent, state_ent
        )

        manager_loss, mets = self._compute_manager_loss(
            imag_feat,
            imag_state,
            imag_action,
            manager_extrinsic_target,
            mgr_ent,
            state_ent,
            weights,
            base,
        )
        metrics.update(mets)

        extrinsic_value_input = imag_feat

        with tools.RequiresGrad(self.manager_exploration_value):
            with torch.cuda.amp.autocast(self._use_amp):
                value = self.manager_exploration_value(extrinsic_value_input[:-1].detach())
                target = torch.stack(manager_extrinsic_target, dim=1)

        with tools.RequiresGrad(self.manager_extrinsic_value):
            with torch.cuda.amp.autocast(self._use_amp):
                value = self.manager_extrinsic_value(extrinsic_value_input[:-1].detach())
                target = torch.stack(manager_extrinsic_target, dim=1)
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = -value.log_prob(target.detach())
                slow_target = self._slow_value_targets["manager_extrinsic"](extrinsic_value_input[:-1].detach())
                if self._config.slow_value_target:
                    value_loss = value_loss - value.log_prob(
                        slow_target.mode().detach()
                    )
                if self._config.value_decay:
                    value_loss += self._config.value_decay * value.mode()
                # (time, batch, 1), (time, batch, 1) -> (1,)
                mgr_extr_value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])
        metrics.update(tools.tensorstats(value.mode(), "manager_extrinsic_value"))
        metrics.update(tools.tensorstats(target, "manager_extrinsic_target"))
        metrics.update(tools.tensorstats(reward_extr, "imag_extr_reward"))


        ## Worker updates
        with tools.RequiresGrad(self.worker):
            with torch.cuda.amp.autocast(self._use_amp):
                pass

        with tools.RequiresGrad(self.worker_value):
            with torch.cuda.amp.autocast(self._use_amp):
                pass

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
            metrics.update(self._manager_opt(manager_loss, self.manager.parameters()))
            metrics.update(self._manager_extr_opt(mgr_extr_value_loss, self.manager_extrinsic_value.parameters()))
        return imag_feat, imag_state, imag_action, weights, metrics
    
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
        goal = self.manager(inp).sample()

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
                goal = self.manager(inp).sample()

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
