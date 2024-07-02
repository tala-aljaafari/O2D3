"""
This file provides the environments with modified noise.

The purpose of this file is to be imported anywhere where the environments are needed.
"""

import numpy as np
from numpy import cos, pi, sin

import gymnasium as gym
from gymnasium.envs.classic_control.acrobot import AcrobotEnv
from gymnasium import logger, spaces
import math
import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.api import VARMAX
from typing import Optional

#for strong noise, ignore warnings on obs being out of the obs space once noise is applied
import warnings
warnings.filterwarnings("ignore")

#seed
np.random.seed(2023)

def VARMAProcess(coeffs, T, k):
    p = 2  # AR order
    model = VARMAX(np.zeros((T, k)), order=(p, 0))

    matrix = np.full((k, k), 0.2)
    np.fill_diagonal(matrix, 0.8)
    matrix[0, 0] = 0.5


    if coeffs[0] == 0.0 and coeffs[1] == 0.95 :
        ar_params = np.hstack((np.zeros((k, k)), matrix))
    elif coeffs[0] == 0.0 and coeffs[1] == 0.0 :
        #return np.zeros((k, T))
        ar_params = np.hstack((np.zeros((k, k)), np.zeros((k, k))))
    else:
        ar_params = np.hstack((matrix, np.zeros((k, k))))

    cov_matrix = np.eye(k)
    lower_tri_indices = np.tril_indices(k)
    cov_matrix = cov_matrix[lower_tri_indices]

    y = model.simulate(params=[0] * k + ar_params.flatten().tolist() + cov_matrix.tolist(), nsimulations=T)
    y = (y-np.mean(y, axis=0)) / np.std(y, axis=0)

    return np.array(y).T



class IMANOAcrobotEnv(AcrobotEnv):
    def __init__(self, 
                 mod_noise_std=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                 train_mod_corr_noise=(0.0, 0.0), 
                 train_noise_strength=1.0, 
                 test_mod_corr_noise=(0.0, 0.0),
                 test_noise_strength=1.0,
                 ep_length=500, 
                 render_mode: Optional[str] = None,
                 step_counter=0,
                 injection_time=0):
        
        super().__init__(render_mode=render_mode)
        self.mod_noise_std = mod_noise_std

        self.train_mod_corr_noise = train_mod_corr_noise
        self.train_noise_strength = train_noise_strength
        self.test_mod_corr_noise = test_mod_corr_noise
        self.test_noise_strength = test_noise_strength

        self.ep_length = ep_length

        self.step_counter = step_counter
        self.injection_time = injection_time

        return
    

    def get_injection_time(self):
        return self.injection_time
        
    def get_episode_length(self):
        return self.ep_length

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):

        #generate the noise variates for the whole episode in advance
        T = self.ep_length + 1
        obs_dim = self.observation_space.shape[0]

        self.train_obs_noise_vec = VARMAProcess(self.train_mod_corr_noise, T, obs_dim) 
        self.test_obs_noise_vec = VARMAProcess(self.test_mod_corr_noise, T, obs_dim) 
        
        self.train_noise_step_ctr = 0
        self.test_noise_step_ctr = 0
        self.step_counter = 0

        ob = super().reset(seed=seed, options=options)

        return ob


    def _get_ob(self):
        s = self.state
        assert s is not None, "Call reset before using AcrobotEnv object."
        ob = np.array(
            [cos(s[0]), sin(s[0]), cos(s[1]), sin(s[1]), s[2], s[3]], dtype=np.float32
            )
        
        if self.step_counter < self.injection_time:
            #Before reaching injection time: inject train noise
            obs_noise = self.train_obs_noise_vec[:, self.train_noise_step_ctr] * self.mod_noise_std * self.train_noise_strength

            noised_ob = np.array(tuple(val + noise for val, noise in zip(ob, obs_noise)), dtype = np.float32)
            self.train_noise_step_ctr += 1
            self.step_counter += 1

        else:
            # When reached injection time: inject test noise
            obs_noise = self.test_obs_noise_vec[:, self.test_noise_step_ctr] * self.mod_noise_std * self.test_noise_strength
            
            noised_ob = np.array(tuple(val + noise for val, noise in zip(ob, obs_noise)), dtype = np.float32)
            self.test_noise_step_ctr += 1

        return noised_ob