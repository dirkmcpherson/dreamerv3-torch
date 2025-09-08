import copy
import torch
from torch import nn
import numpy as np
from PIL import ImageColor, Image, ImageDraw, ImageFont
import random
import time 

import cv2
import networks
import tools
import models as dv3_models

import fastcore.all as fc
import models
import director_networks
import director_utils

# from torchviz import make_dot
from IPython import embed as ipshell

# from uiux import UIUX

to_np = lambda x: x.detach().cpu().numpy()
reshape = lambda tnsr: tnsr.reshape(list(tnsr.shape[:-2]) + [tnsr.shape[-1] ** 2])

class DAutoAdapt(nn.Module):
    def __init__(self, shape) -> None: super().__init__(); fc.store_attr()
    def __call__(self, ent):
        return torch.sum(ent, dim=-1)

class ImaginationActorCritic(nn.Module):
    def __init__(self, config, world_model, input_size, num_actions, value_heads={"reward": 1.0}, stop_grad_actor=True, prefix=''):
        super().__init__()
        self._config = config
        self._wm = world_model
        self._stop_grad_actor = stop_grad_actor
        self._prefix = prefix
        self._config = config


        if self._prefix == "manager":
            actor_act = config.manager_actor['act']
            actor_norm = config.manager_actor['norm']
            actor_dist = config.manager_actor['dist']
            actor_outscale = config.manager_actor['outscale']
            actor_lr = config.manager_lr
            ac_opt_eps = config.manager_opt_eps
            self.actent = lambda x: x # noop 
            # self.actent = tu.AutoAdapt(num_actions[:-1], **config.actent)
            # self.actent = DAutoAdapt(num_actions[:-1]) 
        elif self._prefix == "worker":
            actor_act = config.worker_actor['act']
            actor_norm = 'LayerNorm' # config.worker_actor['norm']
            actor_dist = "onehot" #config.worker_actor['dist']
            actor_outscale = 1.0 #config.worker_actor['outscale']
            actor_lr = config.worker_lr
            ac_opt_eps = 1e-5 #config.ac_opt_eps
            self.actent = lambda x: x # noop 
        else:
            raise NotImplementedError(self._prefix)

        self.actor = director_networks.ActionHead(
            input_size,
            num_actions,
            config.actor_layers,
            config.units,
            actor_act,
            actor_norm,
            actor_dist,
            config.actor_init_std,
            config.actor_min_std,
            config.actor_max_std,
            config.actor_temp,
            outscale=actor_outscale,
            unimix_ratio=config.action_unimix_ratio,
        )
        # self.actor = networks.MLP(
        #     input_size,
        #     num_actions,
        #     config.actor_layers,
        #     config.units,
        #     actor_act,
        #     actor_norm,
        #     actor_dist,
        #     config.actor_init_std,
        #     outscale=actor_outscale,
        #     unimix_ratio=config.action_unimix_ratio,
        #     device=config.device,
        # )
        
        # if value_head == "symlog_disc":
        self.critic_scales = value_heads
        self.critics = nn.ModuleDict(
            {
                key: networks.MLP(
                    input_size,
                    (255,),
                    config.value_layers,
                    config.units,
                    config.act,
                    config.norm,
                    "symlog_disc",
                    outscale=0.1,
                    device=config.device,
                )
                for key in value_heads
            }
        )

        if config.slow_value_target:
            # self._slow_value = copy.deepcopy(self.value)
            self._slow_values = copy.deepcopy(self.critics)
            self._updates = 0
            
        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(),
            lr=actor_lr,
            weight_decay=config.weight_decay,
        )

        self.value_opts = {
                key: torch.optim.Adam(
                    value.parameters(),
                    lr=config.value_lr,
                    weight_decay=config.weight_decay,)
                for key, value in self.critics.items()}

        if use_opt := False:
            pass
            # kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
            # self._actor_opt = tools.Optimizer(
            #     f"{self._prefix}_actor",
            #     self.actor.parameters(),
            #     actor_lr,
            #     ac_opt_eps,
            #     config.actor_grad_clip,
            #     **kw,
            # )
            # self._value_opt = tools.Optimizer(
            #     f"{self._prefix}_value",
            #     self.value.parameters(),
            #     config.value_lr,
            #     ac_opt_eps,
            #     config.value_grad_clip,
            #     **kw,
            # )

        # self.lr_schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(opt, step_size=30, gamma=0.1) for opt in [self.actor_opt, *self.value_opts.values()]]

        if self._config.reward_EMA:
            # self.reward_ema = dv3_models.RewardEMA(device=self._config.device)
            self.reward_emas = {k: dv3_models.RewardEMA(device=self._config.device) for k in self.critics.keys()}
    
    def _train(
        self,
        start=None,
        objective=None,
        repeats=None,
        itraj=None,
    ):
        if itraj == None: raise ValueError("Must provide imag_traj")
        
        self._update_slow_target()
        metrics = {}

        if self._prefix == "manager":
            ifeat = itraj["feat"]
        else:
            ifeat = torch.cat([itraj["feat"], itraj["goal"]], dim=-1)

        istate = itraj["state"]
        iaction = itraj["action"]
        icont = itraj["cont"]
        igoal = itraj["goal"] if "goal" in itraj else None # NOTE: goals become actions for the manager
        iweights = itraj["weight"] if "weight" in itraj else torch.cumprod(self._config.imag_discount * icont, axis=0) / self._config.imag_discount
    
        with tools.RequiresGrad(self.actor):
            state_ent = self._wm.dynamics.get_dist(istate).entropy()

            # NOTE: waste of computation to doulbe calculate weights
            target, mets, weights = self._compute_target(icont, ifeat, itraj); metrics.update(mets)
            
            # NOTE: there's a lot of target regularization in DHafner's version
            actor_loss, mets = self._compute_actor_loss(
                ifeat,
                istate,
                iaction,
                igoal,
                target,
                state_ent,
                weights, # iweights,
            )
            metrics.update(mets)

        value_input = ifeat

        ### Train critics
        '''
        # NOTE: Targets are calculated with the slower target networks, then loss is calculated with the active value networks agaunst the slower targets. The alternate way in models.py (dreamerv3?) is to use the active netowrk for the targets, and then add the negative log-likelihood of the slow network targets. i.e. -value.log_prob(value_target) - value.log_prob(slow_value_target)
        '''
        for k,v in self.critics.items():
            with tools.RequiresGrad(v):
                slow_value_fn = lambda x: self._slow_values[k](x.detach()).mean()
                critic_dist = v(value_input[:-1].detach())
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = -critic_dist.log_prob(v.target.detach()) # NOTE: MSE (critic_dist.mean() - v.target.detach()) ** 2 is the same as gaussian negative log probability (according to hafner)
                # value_loss = -critic_dist.log_prob(target.detach())
                slow_target = slow_value_fn(value_input[:-1])
                if self._config.slow_value_target:
                    value_loss = value_loss - critic_dist.log_prob(
                        slow_target.detach()
                    )

                # (time, batch, 1), (time, batch, 1) -> (1,)
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])
                v.loss = value_loss

                metrics.update(tools.tensorstats(value_loss.detach(), f"{k}_value_loss"))
                metrics.update(tools.tensorstats(v.target, f"{k}_target"))
                metrics.update(tools.tensorstats(slow_target, f"{k}_slow_target"))
                metrics.update(tools.tensorstats(critic_dist.mean().detach(), f"{k}_value_mean"))

        metrics["state_entropy"] = to_np(torch.mean(state_ent))
        with tools.RequiresGrad(self):
            self.actor_opt.zero_grad()
            actor_loss.backward()

            # calculate the gradient norm
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self._config.actor_grad_clip); metrics["actor_grad_norm"] = to_np(grad_norm)

            self.actor_opt.step()

            for k,v in self.critics.items():
                self.value_opts[k].zero_grad()
                v.loss.backward()
                
                # log the gradient norm
                grad_norm = torch.nn.utils.clip_grad_norm_(v.parameters(), self._config.value_grad_clip); metrics[f"{k}_value_grad_norm"] = to_np(grad_norm)

                self.value_opts[k].step()

        
        if self._config.actor_dist in ["onehot"]:
            metrics.update(
                tools.tensorstats(
                    torch.argmax(iaction, dim=-1).float(), "iaction"
                )
            )
        else:
            metrics.update(tools.tensorstats(iaction, "iaction"))
        metrics.update(tools.tensorstats(actor_loss.detach(), "actor_loss"))

        # Name the metrics with the prefix
        prefixxed_metrics = {}
        for k,v in metrics.items():
            if self._prefix not in k:
                prefixxed_metrics[self._prefix+"_"+k] = v
            else:
                prefixxed_metrics[k] = v


        return ifeat, istate, iaction, prefixxed_metrics

    def _compute_target(self, icont, ifeat, itraj):
        metrics = {}
        targets = []
        total = sum([v for k,v in self.critic_scales.items()])
        discount = (self._config.discount * icont).detach() # NOTE: Can this be moved out of the loop with a detach?
        for k,v in self.critics.items():
            value = v(ifeat).mean() * self.critic_scales[k] # NOTE: double use of scalars
            reward = itraj[k]
            score = tools.lambda_return( 
                reward, 
                value[:-1],
                discount[1:],
                bootstrap=value[-1],
                lambda_=self._config.discount_lambda,
                axis=0,
            )

            score = torch.stack(score, dim=1)
            v.target = score.detach() # NOTE: what a mean thing to do to future programmers
            if self._config.reward_EMA:
                base = value[:-1]
                offset, scale = self.reward_emas[k](score)
                normed_target = (score - offset) / scale
                normed_base = (base - offset) / scale

                adv = (normed_target - normed_base) * self.critic_scales[k] / total
                targets.append(adv)
                values = self.reward_emas[k].values; metrics[f"{k}_EMA_005"] = to_np(values[0]); metrics[f"{k}_EMA_095"] = to_np(values[1])
                metrics.update(tools.tensorstats(normed_target, f"{k}_normed_target"));
                # metrics.update(tools.tensorstats(torch.abs(score) >= 0.5).float().mean(), f'{k}_return_rate')
            else: targets.append(score)
            metrics.update(tools.tensorstats(score.detach(), f"{k}_target"))

        target = torch.stack(targets).sum(0)
        metrics.update(tools.tensorstats(target.detach(), "target"))

        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()

        return target, metrics, weights # see note above about lining up value and reward

    def _compute_actor_loss(
        self,
        ifeat,
        istate,
        iaction,
        igoal,
        target,
        state_ent,
        weights,
    ):
        metrics = {}
        # recalculate the entropy to detach it from the graph
        inp = ifeat.detach() if self._stop_grad_actor else ifeat
        policy = self.actor(inp)

        if self._config.imag_gradient == "reinforce":
            logpi = policy.log_prob(iaction.detach())[:-1]

            # adv = (target - self.calc_value(inp[:-1])).detach()
            if self._config.reward_EMA:
                adv = target.detach() # target is advantage 
            else:
                key = "reward" if "reward" in self.critics else "reward_goal"
                adv = (target - self.critics[key](inp[:-1]).mode()).detach()
            loss = -logpi[:, :, None] * adv.detach()
        else:
            raise NotImplementedError(self._config.igradient)
        
        metrics["actor_target"] = to_np(target.detach().mean())
        
        ent = policy.entropy()[:-1]
        ent_loss = self.actent(ent) # noop for worker, autoadapt for manager
        loss -= self._config.actor_entropy * ent_loss[:, :, None]
        loss *= weights[:-1].detach()
        
        metrics.update({f'actent_unweighted': to_np(ent.detach().mean())})
        metrics["actent"] = to_np(torch.mean(ent_loss.detach()))
        
        return loss.mean(), metrics

    def _update_slow_target(self):
        if self._config.slow_value_target:
            if self._updates % self._config.slow_target_update == 0:
                mix = self._config.slow_target_fraction # NOTE: value set from dhafner impl
                for k,v in self.critics.items():
                    for s, d in zip(v.parameters(), self._slow_values[k].parameters()):
                        d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1

class ImaginationAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        fc.store_attr()

        self.goal_enc = networks.MLP(
            config.dyn_deter,
            config.skill_shape, # size of output [8x8]
            **config.goal_encoder
        )
        self.goal_dec = networks.MLP(
            np.prod(config.skill_shape),
            config.dyn_deter,
            **config.goal_decoder
        )
        kw = dict(wd=config.goal_ae_wd, opt=config.opt, use_amp=False)
        self.opt = tools.Optimizer(
            "goal_ae",
            list(self.goal_enc.parameters()) + list(self.goal_dec.parameters()),
            config.goal_ae_lr,
            config.goal_ae_opt_eps,
            config.goal_ae_grad_clip,
            **kw,
        )
        self.skill_prior = torch.distributions.independent.Independent(
            tools.OneHotDist(logits=torch.zeros(*config.skill_shape, device=config.device), unimix_ratio=self.config.action_unimix_ratio), 1
        )

        self.kl_autoadapt = director_utils.AutoAdapt((), **config.kl_autoadapt)


    def _train(self, batch):
        # feat = batch["deter"].detach()
        feat = batch["deter"]
        metrics = {}
        with tools.RequiresGrad(self.goal_enc):
            with tools.RequiresGrad(self.goal_dec):
                enc = self.goal_enc(feat)
                # pre_loss_probs = enc.base_dist.probs.detach().cpu() ## DEBUG
                x = enc.sample() # discrete one-hot grid
                x = x.reshape([x.shape[0], x.shape[1], -1]) # (time, batch, feat0*feat1)
                dec = self.goal_dec(x)
                rec = -dec.log_prob(feat.detach()) # DHafner's original
                if self.config.goal_vae_kl:
                    # NOTE: Should kl and recreation loss be scaled by scheduled constants as in the world model?
                    kl = kl_nonorm = torch.distributions.kl.kl_divergence(enc, self.skill_prior)
                    if self.config.goal_vae_kl_autoadapt:
                        kl, mets = self.kl_autoadapt(kl_nonorm)
                        metrics.update({f'goalkl_{k}':to_np(v) for k, v in mets.items()})
                    else:
                        # simple kl adaptations
                        kl *= self.config.goal_vae_kl_beta
                    metrics.update({'goalkl_nonorm': to_np(kl_nonorm.mean())})
                    assert kl.shape == rec.shape, (kl.shape, rec.shape)
                    loss = torch.mean(rec + kl)
                else:
                    loss = torch.mean(rec)
                
                metrics.update(self.opt(loss, list(self.goal_enc.parameters()) + list(self.goal_dec.parameters())))
        return metrics

class ImaginationAgent(nn.Module):
    def __init__(self, config, logger, worker: ImaginationActorCritic, manager: ImaginationActorCritic, goalae: ImaginationAE, _wm: models.WorldModel, dataset, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        fc.store_attr()
        self._step = director_utils.count_steps(self.config.traindir); self._train_steps = 0; 
        self.train_every = director_utils.Ratio(self.config.train_ratio / (self.config.batch_size * self.config.batch_length)) #self.train_period = 2;
        self.log_period = 1e3; self.log_vid_period = 3e3
        self._pretrained = False; self._metrics = {}; 
        self.logger.step = self._step
        # self.uiux = UIUX()
        # self.uiux.slider.set(int(config.skill_alpha * 100))
        # self.uiux.update_text(n_clusters=config.n_human_clusters, n_samples=config.n_skill_samples)

    def __call__(self, obs, done, state, training=True, human=False):
        if state is not None and done.any(): # zero out the state if we're done
            mask = 1 - done
            for key in state[0].keys():
                for i in range(state[0][key].shape[0]):
                    state[0][key][i] *= mask[i]
            for i in range(len(state[1])):
                state[1][i] *= mask[i]

        if training:
            if self.config.pretrain and not self._pretrained:
                self._pretrained = True
                for i in range(self.config.pretrain):
                    # self._train(next(self.dataset)); self._train_steps += 1; self._step +=1; self.logger.step = self._step; self._log()
                    self._pretrain(next(self.dataset)); self._train_steps += 1; self._step +=1; self.logger.step = self._step; self._log()
                    # Insert "steps" here just to see how the distributions change over pretraining

            # elif self._step % self.train_period == 0:
            elif self.train_every(self._step):
                self._train(next(self.dataset)); self._train_steps += 1

            if self._step % self.log_period == 0: self._log()
            if self._step % self.log_vid_period == 0: self._log_video()

        policy_output, state = self._policy(obs, state, training, human)
        self._step += 1
        self.logger.step = self._step
        
        if self.config.human and False:
            latent, action, goal = state
            goal_stoch = self._wm.dynamics.get_stoch(goal) # latent["stoch"]
            toshow = obs["image"][0].copy(); inp = torch.cat([reshape(goal_stoch), goal], dim=-1)
            goal_img = self._wm.heads["decoder"](inp[None])["image"].mode().squeeze().detach().cpu().numpy() if state is not None else np.ones_like(obs["image"][0].copy())
            toshow = np.concatenate([toshow, goal_img], axis=1); toshow = cv2.resize(toshow, (0,0), fx=3, fy=3)
            cv2.imshow('live', cv2.putText(toshow, f"#{self._step}/{self._train_steps}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))); cv2.waitKey(1)

        return policy_output, state
    
    def _policy(self, obs, state, training=False, human=False):
        if state is None:
            batch_size = len(obs["image"])
            latent = self._wm.dynamics.initial(len(obs["image"]))
            action = torch.zeros((batch_size, self.config.num_actions)).to(self.config.device)
            goal = None
        else:
            latent, action, goal = state

        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(
            latent, action, embed, obs["is_first"], self.config.collect_dyn_sample
        )
        feat = self._wm.dynamics.get_feat(latent)

        # if self.uiux:
        #     if (self.uiux.expert_model is not None):
        #         with torch.no_grad():
        #             prev_exp_latent = self.uiux.expert_model.latent if hasattr(self.uiux.expert_model, "latent") else self._wm.dynamics.initial(len(obs["image"]))
        #             exp_embed = self.uiux.expert_model._wm.encoder(obs)
        #             exp_latent, _ = self.uiux.expert_model._wm.dynamics.obs_step(prev_exp_latent, action, exp_embed, obs["is_first"], self.config.collect_dyn_sample)
        #             self.uiux.expert_model.latent = exp_latent # sorry for the hack
        #     elif human:
        #         self.uiux.update_feed(obs["image"])


        sample = training
        # use_human_skill_duration = human and (self.uiux.expert_model is None)
        # skill_duration = self.config.train_skill_duration if not use_human_skill_duration else self.config.human_i_horizon # for humans run just short of the depicted rollouts
        skill_duration = self.config.imag_horizon - 2 #self.config.train_skill_duration
        if goal == None or self._step % skill_duration == 0:
            manager_chooses_skill = not human
            if human:
                t0 = time.time()
                # skill, goal_deter, exit_code, metrics = hri.human_interaction(latent, self, obs, self.config.n_skill_samples, sample=False); self._metrics.update(metrics)
                if exit_code < 0:
                    print(f"Human selected goal with exit code {exit_code}.")
                    manager_chooses_skill = True
                # print(f"hri took {time.time() - t0:1.2f} seconds.")

            if manager_chooses_skill: skill, goal_deter, goal_stoch, img = self.get_goal(feat, sample=sample, gen_image=False)
            goal = goal_deter

        inp = torch.cat([feat, goal], dim=-1)
        actor_dist = self.worker.actor(inp)
        # NOTE: The actor really should act stochastically with sample until it's settled into good policies. This needs to be fixed with human.
        action = actor_dist.sample() if sample else actor_dist.mode()

        # detach the latent state so we don't backpropagate through it
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()        

        log_prob = actor_dist.log_prob(action)        

        state = (latent, action, goal)
        policy_output = {"action": action, "logprob": log_prob}
        return policy_output, state
    

    def _pretrain(self, batch):
        # just pretrain the world model and the goalae
        post, prior, metrics = self._wm._train(batch); self._metrics.update(metrics)
        self._metrics.update(self.goalae._train(post))

    def _train(self, batch):
        post, prior, metrics = self._wm._train(batch); self._metrics.update(metrics)
        self._metrics.update(self.goalae._train(post))

        with tools.RequiresGrad(self.worker.actor):
            ifeats, istates, iactions, iskills, igoals = director_utils.imagine_with_skills(self.config, self._wm.dynamics, post, self.worker.actor, self.manager.actor, self.goalae.goal_dec, self.config.imag_horizon+1)
        feats = self._wm.dynamics.get_feat(istates) # use start plus successor states
        icont = self._wm.heads["cont"](feats).mean
        ireward = self._wm.heads["reward"](feats).mean()[1:] # or mode?
        itraj = {
            # "image": images,
            "stoch": istates["stoch"].detach(),
            "deter": istates["deter"].detach(),
            "logit": istates["logit"].detach(),
            "feat": ifeats.detach(),
            "action": iactions.detach(),
            "reward": ireward.detach(),
            "skill": iskills.detach(),
            "goal": igoals.detach(),
            "cont": icont.detach(),
        }
        mtraj = self.abstract_traj(itraj, deep_copy_input=True)
        mtraj["state"] = {"stoch": mtraj["stoch"], "deter": mtraj["deter"], "logit": mtraj["logit"]}
        mtraj["reward_expl"] = self.expl_reward(mtraj["state"])[1:][..., None]
        *_, metrics = self.manager._train(itraj=mtraj); self._metrics.update(metrics)

        wtraj = self.split_traj(itraj, deep_copy_input=True)
        # wtraj = self.split_traj(itraj)
        wtraj["state"] = {"stoch": wtraj["stoch"], "deter": wtraj["deter"], "logit": wtraj["logit"]}
        wtraj["reward_goal"] = self.goal_reward(wtraj["state"], wtraj["goal"])[1:][..., None]
        *_, metrics = self.worker._train(itraj=wtraj); self._metrics.update(metrics)
        # *_, metrics = self.worker._train(itraj=itraj); self._metrics.update(metrics)

    def _log(self):
        self._metrics["update_count"] = self._train_steps
        for k, v in self._metrics.items():
            try:
                self.logger.scalar(k, np.mean(v))
            except:
                print(f"Could not log {k}, {v.shape if hasattr(v, 'shape') else None}")
        self._metrics = {}
        self.logger.write()

    def _log_video(self):
        normalize_image = lambda img: (img - img.min()) / (img.max() - img.min())
        batch = next(self.dataset)
        openl, openl_state = self._wm.video_pred(batch)
        feat = self._wm.dynamics.get_feat(openl_state)
        _, goal_deter, goal_stoch, _ = self.get_goal(feat)
        inp = torch.cat([reshape(goal_stoch), goal_deter], -1)
        gimg = self._wm.heads["decoder"](inp)["image"].mode().squeeze().detach().cpu().numpy()

        # we actually need every eighth frame, duplicated 8 times
        n = self.config.train_skill_duration; repeatn = gimg.shape[1] // n; 
        gimg = np.repeat(gimg[:, ::n], repeatn, axis=1)
        openl = np.concatenate([to_np(openl), normalize_image(gimg)], axis=2)
        self.logger.video("train_openl", openl)

        # see how the goalae is doing
        goalae_enc = self.goalae.goal_enc(openl_state["deter"]).sample()
        goalae_deter = self.goalae.goal_dec(reshape(goalae_enc)).mode(); goal_stoch = self._wm.dynamics.get_stoch(goalae_deter)
        goalae_img = self._wm.heads["decoder"](torch.cat([reshape(goal_stoch), goal_deter], dim=-1))["image"].mode().squeeze().detach().cpu().numpy()
        normed_images = normalize_image(batch["image"][:6])
        self.logger.video("goalae_openl", np.concatenate([normed_images, normalize_image(goalae_img)], axis=2))

        n = self.config.train_skill_duration; repeatn = normed_images.shape[1] // n
        gfeat = torch.repeat_interleave(feat[:, ::n, :], repeatn, dim=1)
        skills = self.manager.actor(gfeat).sample(); skills = reshape(skills)
        goals = self.goalae.goal_dec(skills).mode(); goal_stoch = self._wm.dynamics.get_stoch(goals)
        goal_imgs = self._wm.heads["decoder"](torch.cat([reshape(goal_stoch), goal_deter], dim=-1))["image"].mode().squeeze().detach().cpu().numpy()
        self.logger.video("realtop_goalbot", np.concatenate([normed_images, normalize_image(goal_imgs)], axis=2))

    def get_goal(self, feat, sample=True, gen_image=False):
        skill = self.manager.actor(feat).sample() if sample else self.manager.actor(feat).mode()
        goal_deter = self.goalae.goal_dec(reshape(skill)).mode()
        goal_stoch = self._wm.dynamics.get_stoch(goal_deter)
        img = None
        if gen_image:
            inp = torch.cat([reshape(goal_stoch), goal_deter], -1)
            # pad the tensor shape
            # print(inp.shape)
            img = self._wm.heads["decoder"](inp[None, ...])["image"].mode().squeeze().detach().cpu().numpy()

        return skill, goal_deter, goal_stoch, img

    def split_traj(self, orig_traj, deep_copy_input=True):
        traj = director_utils.deep_copy_dict(orig_traj) if deep_copy_input else orig_traj
        k = self.config.train_skill_duration
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

        if 'goal' in traj: traj['goal'] = torch.concat([traj['goal'][:-1], traj['goal'][:1]], 0)

        traj['weight'] = torch.cumprod(
            self.config.imag_discount * traj['cont'], axis=0) / self.config.imag_discount
        return traj
    
    ''' NOTE:
    These two deep copys use a huge amount of memory. Is there a way to combine them?
    '''

    def abstract_traj(self, orig_traj, deep_copy_input=True):
        traj = director_utils.deep_copy_dict(orig_traj) if deep_copy_input else orig_traj
        traj["action"] = traj.pop("skill")
        k = self.config.train_skill_duration

        if self.config.unfold_manager_traj:
            # unfold into every length k subtrajectory
            def reshape(x):
                x = x.unfold(0, k, 1) # (n, ..., k)
                return x.reshape([x.shape[0], k, *x.shape[1:-1]]) # reshape to match (n, k, ...)
        else: reshape = lambda x: x.reshape([x.shape[0] // k, k] + list(x.shape[1:]))

        weights = torch.cumprod(reshape(traj["cont"][:-1]), 1).to(self.config.device)
        for key, value in list(traj.items()):
            if 'reward' in key:
                # if (traj[key] > 1.0).any():
                #     a = 1
                # NOTE: Paper says to use the sum, DHafner code uses the mean
                # traj[key] = (reshape(value) * weights).mean(1) # weighted mean of each subtrajectory
                traj[key] = (reshape(value) * weights).sum(1) # weighted sum of each subtrajectory
            elif key == 'cont':
                traj[key] = torch.concat([value[:1], reshape(value[1:]).prod(1)], 0)
            else:
                traj[key] = torch.concat([reshape(value[:-1])[:, 0], value[-1:]], 0)
        traj['weight'] = torch.cumprod(
            self.config.imag_discount * traj['cont'], 0) / self.config.imag_discount
        
        ashape = traj["action"].shape
        traj["action"] = traj["action"].reshape(*ashape[:len(ashape)-1], *self.config.skill_shape)
        return traj
    
    def goal_reward(self, state, goal):
        # cosine_max reward
        h = state["deter"]
        goal = goal.detach()
        gnorm = torch.linalg.norm(goal, dim=-1, keepdim=True) + 1e-12
        hnorm = torch.linalg.norm(h, dim=-1, keepdim=True) + 1e-12
        norm = torch.max(gnorm, hnorm)
        return torch.einsum("...i,...i->...", goal / norm, h / norm) # NOTE

    def expl_reward(self, state):
        # elbo (Evidence Lower BOund) reward
        feat = state["deter"]
        enc = self.goalae.goal_enc(feat) 
        x = enc.sample() # produces a skill_dim x skill_dim
        x = x.reshape([x.shape[0], x.shape[1], -1]) # (time, batch, feat0*feat1)
        dec = self.goalae.goal_dec(x) # decode back the feature dimensions

        return ((dec.mode() - feat) ** 2).mean(-1) # NOTE: This can probably be a log likelihood under the MSE distribution

    def save(self, logdir, note=''):
        torch.save(self.state_dict(), logdir / f"model{note}.pt")

    def load(self, logdir, note=''):
        self.load_state_dict(torch.load(logdir / f"model{note}.pt"))