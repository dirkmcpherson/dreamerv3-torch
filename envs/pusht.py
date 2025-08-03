import gymnasium as gym
import gym_pusht as pusht
import numpy as np
import cv2

class PushT(gym.Env):
    def __init__(self, size=(64,64), obs_type="pixels_state", render_mode="rgb_array", force_sparse=False, max_steps=1000, action_repeat=1):
        w,h = size
        self._env = gym.make("gym_pusht/PushT-v0", obs_type=obs_type, render_mode=render_mode, observation_width=w, observation_height=h, force_sparse=force_sparse, display_cross=False)
        self._obs_is_dict = hasattr(self._env.observation_space, "spaces")
        self._obs_key = "image"
        self.force_sparse = force_sparse # Force the reward to be sparse
        self.max_steps = max_steps; self.nstep = 0
        self.action_repeat = action_repeat

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)
        
    def step(self, action):
        
        for _ in range(self.action_repeat):
            obs, reward, done, truncated, info = self._env.step(action)
            if done: break

        if not self._obs_is_dict:
            obs = {self._obs_key: obs}
        else:
            for k,v in obs.items():
                obs[k] = np.array(v)

        if "image" not in obs and "pixels" in obs:
            obs["image"] = obs["pixels"]
        if 'pixels' in obs: obs.pop('pixels')

        info['success'] = np.array(info.get('is_success', False))
        info['coverage'] = np.array(info.get('coverage', 0.0))

        if info["is_success"]:
            reward = 2 * self.max_steps # self.max_steps - self.nstep
            print("Success!")

        if self.force_sparse:
            reward = 1.0 if info['is_success'] else 0.0

        # Transpose 
        obs["is_first"] = False
        obs["is_last"] = done
        obs["is_terminal"] = info.get("is_terminal", False)

        self.nstep += 1
        return obs, reward, done, info

    @property
    def observation_space(self):
        if self._obs_is_dict:
            spaces = self._env.observation_space.spaces.copy()
        else:
            spaces = {self._obs_key: self._env.observation_space}

        # replace pixels with image
        if "pixels" in spaces:
            spaces["image"] = spaces.pop("pixels")
            
        return gym.spaces.Dict(
            {
                **spaces,
                "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
                "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
                "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            }
        )
    
    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    def reset(self):
        obs, info = self._env.reset()
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}

        if "image" not in obs and "pixels" in obs:
            obs["image"] = obs["pixels"]

        obs["is_first"] = True
        obs["is_last"] = False
        obs["is_terminal"] = False
        self.nstep = 0
        return obs

    def render(self, mode="human"):
        return self._env.render()

    def close(self):
        self._env.close()

if __name__ == '__main__':
    env = PushT(max_steps=300)
    env.reset()
    a = env.render()
    print(np.max(a), np.min(a))
        