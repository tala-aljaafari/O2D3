import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.do_render = False
        self.prev_qpos = None
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, "%s/assets/half_cheetah.xml" % dir_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        self.prev_qpos = np.copy(self.sim.data.qpos.flat)
        action = action.reshape(
            6,
        )
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()

        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = ob[0] - 0.0 * np.square(ob[2])
        reward = reward_run + reward_ctrl

        done = False
        if self.do_render:
            self.viewer.cam.lookat[0] = self.sim.data.qpos.flat[0]
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate(
            [
                (self.sim.data.qpos.flat[:1] - self.prev_qpos[:1]) / self.dt,
                self.sim.data.qpos.flat[1:],
                self.sim.data.qvel.flat,
            ]
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.normal(loc=0, scale=0.001, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.normal(loc=0, scale=0.001, size=self.model.nv)
        self.set_state(qpos, qvel)
        self.prev_qpos = np.copy(self.sim.data.qpos.flat)
        return self._get_obs()

    def viewer_setup(self):
        self.do_render = True
        self.viewer.cam.distance = self.model.stat.extent * 0.25
        self.viewer.cam.elevation = -55

from .anom_mj_env import AnomMJEnv
class AnomHalfCheetahEnv(AnomMJEnv, HalfCheetahEnv):
    pass