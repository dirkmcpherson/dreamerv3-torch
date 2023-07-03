import argparse
import collections
import functools
import os
import pathlib
import sys
import warnings

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import director_models
import models
import tools
import envs.wrappers as wrappers

import torch
from torch import nn
from torch import distributions as torchd

from IPython import embed as ipshell
from uiux import UIUX, CirclePattern

to_np = lambda x: x.detach().cpu().numpy()


class Director(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super(Director, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        self._step = count_steps(config.traindir)
        self._update_count = 0
        # Schedules.
        config.actor_entropy = lambda x=config.actor_entropy: tools.schedule(
            x, self._step
        )
        config.actor_state_entropy = (
            lambda x=config.actor_state_entropy: tools.schedule(x, self._step)
        )
        config.imag_gradient_mix = lambda x=config.imag_gradient_mix: tools.schedule(
            x, self._step
        )
        self._dataset = dataset
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = director_models.HierarchyBehavior(
            config, self._wm, config.behavior_stop_grad
        )
        if config.compile:
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
            # self._alt_behavior = torch.compile(self._alt_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

    def __call__(self, obs, reset, step_in_ep, state=None, reward=None, training=True, human=False):
        assert not (human and training) # can't be both for now
        step = self._step
        if self._should_reset(step):
            state = None
        if state is not None and reset.any():
            mask = 1 - reset
            for key in state[0].keys():
                for i in range(state[0][key].shape[0]):
                    state[0][key][i] *= mask[i]
            for i in range(len(state[1])):
                state[1][i] *= mask[i]
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
                print(f"\ttrain_step: {self._update_count} at {step}")
            if self._should_log(step) or self._config.debug_always_update:
                print(f"\tlog_step: {self._update_count}")
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []

                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))

                    goal_vid_pred = self._task_behavior.goal_video_pred(next(self._dataset))
                    self._logger.video("train_goal_vid_pred", to_np(goal_vid_pred))

                    # truth, model, error = self._alt_behavior.goal_pred(next(self._dataset))
                    goal_pred = self._task_behavior.goal_pred(next(self._dataset))
                    

                    self._logger.batch_images("train_goal", to_np(goal_pred))

                # self._logger.add_graph(self._task_behavior.goal_enc, torch.zeros((1, 3, 1024)).to(self._config.device))
                # self._logger.add_graph(self._task_behavior.goal_dec, torch.zeros((1, 3, 64)).to(self._config.device))
                self._logger.write(fps=True)
                print(f"\t\tlog_step: {self._update_count} done")

        policy_output, state = self._policy(obs, state, step_in_ep, training, human)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step

        if human:
            # gotta do something to mark the trajectories the human is creating.
            pass

        return policy_output, state

    def _policy(self, obs, state, step_in_ep, training, human=False):
        if state is None:
            batch_size = len(obs["image"])
            latent = self._wm.dynamics.initial(len(obs["image"]))
            action = torch.zeros((batch_size, self._config.num_actions)).to(self._config.device)
            goal_dim = self._config.dyn_stoch * self._config.dyn_discrete # NOTE: This should be deter I think and it's a coincidence this is working
            goal = torch.zeros((batch_size, goal_dim)).to(self._config.device)
        else:
            latent, action, goal = state

        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(
            latent, action, embed, obs["is_first"], self._config.collect_dyn_sample
        )
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        
        # is it time to select a new goal?
        if human:
            img = (obs["image"].detach().cpu().numpy().squeeze()+0.5) * 255
            action_str = {0: "noop", 1: "forward", 2: "left", 3: "right", 4: "forward_left", 5: "forward_right"}[action.detach().cpu().numpy().squeeze().argmax()]

            obs_string = f"{step_in_ep} {action_str}, reward: {obs['reward'].detach().cpu().numpy().squeeze():1.2f}"
            self._task_behavior.uiux.update_obs(img, obs_string)
            print(f"Updating obs at step {step_in_ep}. {obs_string}")

        if step_in_ep % self._config.train_skill_duration == 0:
            # print(f"Selecting a new goal because step {step_in_ep} % {self._config.train_skill_duration} == 0")
            # NOTE: We should keep a count that starts from 0 when tasks start
            _, goal = self._task_behavior.sample_goal(latent, human=human)
        
        gc_feat = torch.cat([feat, goal], dim=-1)

        # print(f"Director::_policy"); ipshell()

        if not training:
            actor = self._task_behavior.worker.actor(gc_feat)
            action = actor.mode()
            # actor = self._task_behavior.actor(feat)
            # action = actor.mode()

            # worker = self._alt_behavior.worker.actor(gc_feat)
            # worker_action = worker.mode()

        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(gc_feat) if hasattr(self._expl_behavior, "actor") else self._expl_behavior.worker.actor(gc_feat)
            action = actor.sample()

            # worker = self._expl_behavior.worker(gc_feat)
            # action = worker.sample()
        else:
            actor = self._task_behavior.worker.actor(gc_feat)
            action = actor.sample()
            # actor = self._task_behavior.actor(feat)
            # action = actor.sample()

            # worker = self._alt_behavior.worker.actor(gc_feat)
            # worker_action = worker.sample()

        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action, goal)
        return policy_output, state

    def _train(self, data):
        metrics = {}
        if self._config.debug:
            for k,v in data.items():
                print(f"{k} {v.shape if hasattr(v, 'shape') else 'no shape'}")
        post, wm_out, mets = self._wm._train(data)
        metrics.update(mets)
        context = {**data, **post, **wm_out}
        start = post

        metrics.update(self._task_behavior._train(start, context)[-1])

        # metrics.update(self._alt_behavior._train(start)[-1])
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, wm_out, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)

from dreamer import make_env, make_dataset, count_steps, ProcessEpisodeWrap

def main(config):
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    logger = tools.Logger(logdir, config.action_repeat * step)

    print("Create envs.")
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    make = lambda mode: make_env(config, logger, mode, train_eps, eval_eps)
    train_envs = [make("train") for _ in range(config.envs)]
    eval_envs = [make("eval") for _ in range(config.envs)]
    acts = train_envs[0].action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    if not config.offline_traindir:
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        if hasattr(acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.Tensor(acts.low).repeat(config.envs, 1),
                    torch.Tensor(acts.high).repeat(config.envs, 1),
                ),
                1,
            )

        def random_agent(o, d, s, r):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        tools.simulate(random_agent, train_envs, prefill)
        logger.step = config.action_repeat * count_steps(config.traindir)

    print("Simulate agent.")
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)

    # NOTE: JS, bad code
    if config.show_trained_policy:
        print(f"Forcing CPU device to show trained policy.")
        config.device = "cpu"

    agent = Director(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if (logdir / "latest_model.pt").exists():
        print(f"Loding pre-trained model from {logdir / 'latest_model.pt'}")
        agent.load_state_dict(torch.load(logdir / "latest_model.pt"))
        agent._should_pretrain._once = False

        #### JS: load the pre-trained policy and mess with it
        if config.show_trained_policy:
            import time
            import cv2
            print(f"Displaying the trained policy and then exiting..")
            # classic RL loop
            env = eval_envs[0]
            obs = env.reset()
            state = None
            obs["reward"] = torch.Tensor([obs["reward"]])
            obs["is_first"] = torch.Tensor([obs["is_first"]])
            obs["is_last"] = torch.Tensor([obs["is_last"]])
            obs["is_terminal"] = torch.Tensor([obs["is_terminal"]])

            # wrap everything in tensors
            # for k, v in obs.items():
            #     obs[k] = torch.tensor(v, dtype=torch.float32).unsqueeze(0)
            for ep in range(10):
                done = np.array([False])
                step_in_ep = 0
                while not any(done):
                    # for k, v in obs.items():
                    #     obs[k] = torch.tensor(v, dtype=torch.float32)
                    obs["is_terminal"] = torch.Tensor(obs["is_terminal"])
                    obs["image"] = torch.Tensor(obs["image"]).unsqueeze(0)
                    action, state = agent(obs, done, step_in_ep=step_in_ep, state=state, training=False, human=False)
                    action["action"] = to_np(action["action"])[0]
                    obs, reward, done, info = env.step(action)
                    # print("obs", obs)
                    # display the image
                    if False: # or hasattr(env, "render"):
                        env.render()
                    else:
                        print(f"Episode {ep} reward: {reward}, done: {done}")
                        # display with opencv
                        # blow up the image
                        latent, action, goal = state
                        stoch = agent._wm.dynamics.get_stoch(goal)
                        stoch = stoch.reshape(1, -1)
                        inp = torch.cat([stoch, goal], axis=-1)

                        goal_img = agent._wm.heads["decoder"](inp.unsqueeze(0))["image"].mode()
                        goal_img = to_np(goal_img.squeeze(0).squeeze(0))

                        img = np.concatenate([obs["image"], goal_img], axis=0)
                        # ipshell()
                        # img = cv2.resize(img, (1024, 512))
                        scaled_img = cv2.resize(obs["image"], (512, 512))
                        cv2.imshow("image", scaled_img)
                        cv2.imshow("alt", img)
                        cv2.imshow("goal", goal_img)
                        cv2.waitKey(1)
                    time.sleep(0.01)
                    done = np.array([done])
                    step_in_ep += 1
            exit()
        #############################


    state = None
    if config.debug_uiux:
        user_steps = 0
        while user_steps < config.debug_user_steps:
            logger.write()
    else:
        loops = 0
        while agent._step < config.steps:
            logger.write()
            # if config.only_human_policy:
            #     print("Only running human policy, then quitting")
            #     human_policy = functools.partial(agent, training=False, human=True)
            #     state = tools.simulate(human_policy, train_envs, config.eval_every, state=state, include_episode_step=True)
            #     break

            print("----------evaluation-------------")
            eval_policy = functools.partial(agent, training=False)
            tools.simulate(eval_policy, eval_envs, episodes=config.eval_episode_num, include_episode_step=True)
            
            if config.video_pred_log:
                video_pred = agent._wm.video_pred(next(eval_dataset))
                logger.video("eval_openl", to_np(video_pred))
                print(f"\tFinished eval video prediction.")

            print("-----------training--------------")
            state = tools.simulate(agent, train_envs, steps=config.eval_every, state=state, include_episode_step=True)

            if config.human_policy and (loops % config.human_train_interval) == 0 and (config.human_skip_first_train or loops > 0):
                ''' 
                interactive mode
                '''
                print("-----------human--------------")
                human_policy = functools.partial(agent, training=False, human=True)
                state = tools.simulate(human_policy, train_envs, steps=config.human_steps, state=state, include_episode_step=True)
            torch.save(agent.state_dict(), logdir / "latest_model.pt")
            loops += 1

    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    config = parser.parse_args(remaining)
    main(config)
