
import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.api import VARMAX

def VARMAProcess(coeffs, T, k):
    # Define a VARMA model of order (2,0)
    p = 2  
    model = VARMAX(np.zeros((T, k)), order=(p, 0))

    # Correlation matrix 
    # diagonal is 0.8, first elt is 0.5, all other elts are 0.2
    matrix = np.full((k, k), 0.2)
    np.fill_diagonal(matrix, 0.8)
    matrix[0, 0] = 0.5

    # 1-step correlation
    if coeffs[0] == 0.95 and coeffs[1] == 0.0 : 
        ar_params = np.hstack((matrix, np.zeros((k, k))))
    # 2-step correlation
    elif coeffs[0] == 0.0 and coeffs[1] == 0.95 :
        ar_params = np.hstack((np.zeros((k, k)), matrix))
    # no correlation
    else:
        #return np.zeros((k, T)) 
        ar_params = np.hstack((np.zeros((k, k)), np.zeros((k, k))))

    # Covariance matrix for epsilon
    cov_matrix = np.eye(k)
    lower_tri_indices = np.tril_indices(k)
    cov_matrix = cov_matrix[lower_tri_indices]

    # The parameters of the model have to be added in a weird way ..
    y = model.simulate(params=[0] * k + ar_params.flatten().tolist() + cov_matrix.tolist(), nsimulations=T)
    y = (y-np.mean(y, axis=0)) / np.std(y, axis=0)

    return np.array(y).T

class AnomMJEnv:
    """
    Based on Haider's AnomMJEnv
    """

    def __init__(
        self,
        bm_factor_train=1,
        bm_factor_test=1,
        force_vec_dict_train={},
        force_vec_dict_test={},
        act_factor=None,
        act_offset=None,
        act_noise=None,
        obs_factor=None,
        obs_offset=None,
        obs_noise_train=None,
        obs_noise_test=None,
        state_noise_train=None,
        state_noise_test=None,
        force_noise_train=None,
        force_noise_test=None,
        injection_time=100,
    ):
        self.step_counter = 0
        self.injection_time = injection_time
        self.force_vec_dict_train = force_vec_dict_train
        self.force_vec_dict_test = force_vec_dict_test
        self.act_factor = act_factor
        self.act_offset = act_offset
        self.act_noise = act_noise
        self.obs_factor = obs_factor
        self.obs_offset = obs_offset
        self.obs_noise_train = obs_noise_train
        self.obs_noise_test = obs_noise_test
        self.state_noise_train = state_noise_train
        self.state_noise_test = state_noise_test
        self.force_noise_train = force_noise_train
        self.force_noise_test = force_noise_test
        super().__init__()
        self.nominal_bm = self.model.body_mass.copy()
        self.bm_factor_train = bm_factor_train
        self.bm_factor_test = bm_factor_test

    def dist_parameters_train(self):
        if hasattr(self, "nominal_bm"):
            self.model.body_mass[:] = self.nominal_bm[:] * self.bm_factor_train
        for body_name, f_vec in self.force_vec_dict_train.items():
            train_noise_strength = self.force_noise_train["noise_strength"]
            body_id = self.sim.model.body_name2id(body_name)
            if hasattr(self, "train_force_noise_vec"):
                self.sim.data.xfrc_applied[body_id][:] = f_vec[:] * self.train_force_noise_vec[:, self.step_counter] * train_noise_strength

    def dist_parameters_test(self):
        self.model.body_mass[:] = self.nominal_bm[:] * self.bm_factor_test
        for body_name, f_vec in self.force_vec_dict_test.items():
            test_noise_strength = self.force_noise_test["noise_strength"]
            body_id = self.sim.model.body_name2id(body_name)
            self.sim.data.xfrc_applied[body_id][:] = f_vec[:] * self.test_force_noise_vec[:, self.step_counter] * test_noise_strength

    def reset_parameters(self):
        self.model.body_mass[:] = self.nominal_bm[:]
        self.force_applied = np.zeros_like(self.sim.data.xfrc_applied)

    def noise_train(self, obs):
        if self.obs_noise_train is not None:
            train_noise_strength = self.obs_noise_train["noise_strength"]
            obs_noise = self.train_obs_noise_vec[:, self.step_counter] * train_noise_strength
            obs = tuple(val + noise for val, noise in zip(obs, obs_noise))
        return obs

    def noise_test(self, obs):
        if self.obs_noise_test is not None:
            test_noise_strength = self.obs_noise_test["noise_strength"]
            obs_noise = self.test_obs_noise_vec[:, self.step_counter] * test_noise_strength
            obs = tuple(val + noise for val, noise in zip(obs, obs_noise))
        return obs

    def step(self, act):

        if self.step_counter < self.injection_time:
            self.dist_parameters_train()
            self.step_counter += 1

            if self.state_noise_train is not None:
                sim_state = self.sim.get_state()
                train_noise_strength = self.state_noise_train["noise_strength"]
                state_noise = self.train_state_noise_vec[:, self.step_counter] * train_noise_strength
                sim_state.qpos[:] = tuple(val + noise for val, noise in zip(sim_state.qpos, state_noise))
                self.sim.set_state(sim_state)
                self.sim.forward()
            obs, reward, done, info = super().step(act)
            obs = self.noise_train(obs)
            return obs, reward, done, info

        else:
            if self.step_counter == self.injection_time:
                self.dist_parameters_test()

            if self.act_offset is not None:
                act = act + self.act_offset
            if self.act_factor is not None:
                act = act * self.act_factor
            if self.act_noise is not None:
                act = np.random.normal(act, self.act_noise)

            if self.state_noise_test is not None:
                sim_state = self.sim.get_state()
                test_noise_strength = self.state_noise_test["noise_strength"]
                state_noise = self.test_state_noise_vec[:, self.step_counter] * test_noise_strength
                sim_state.qpos[:] = tuple(val + noise for val, noise in zip(sim_state.qpos, state_noise))
                self.sim.set_state(sim_state)
                self.sim.forward()
            obs, reward, done, info = super().step(act)

            if self.obs_offset is not None:
                obs = obs + self.obs_offset
            if self.obs_factor is not None:
                obs = obs * self.obs_factor

            obs = self.noise_test(obs)

            self.step_counter += 1
            return obs, reward, done, info

    def set_seed(self, env_seed):
        self.env_seed = env_seed

    def reset(self):
        self.seed(self.env_seed)
        self.reset_parameters()

        T = self.spec.max_episode_steps + 1 # requires TimeLim wrapper!

        _ = super().reset()  # making sure we have a state!
        obs_dim = len(self.observation_space.low)
        state_dim = len(self.sim.get_state().qpos)
        force_dim = 1

        if self.obs_noise_train is not None:
            obs_mod_corr_noise_train = self.obs_noise_train["mod_corr_noise"]
            self.train_obs_noise_vec = VARMAProcess(obs_mod_corr_noise_train, T, obs_dim)
        if self.obs_noise_test is not None:
            obs_mod_corr_noise_test = self.obs_noise_test["mod_corr_noise"]
            self.test_obs_noise_vec = VARMAProcess(obs_mod_corr_noise_test, T, obs_dim)

        if self.state_noise_train is not None:
            state_mod_corr_noise_train = self.state_noise_train["mod_corr_noise"]
            self.train_state_noise_vec = VARMAProcess(state_mod_corr_noise_train, T, state_dim)
        if self.state_noise_test is not None:
            state_mod_corr_noise_test = self.state_noise_test["mod_corr_noise"]
            self.test_state_noise_vec = VARMAProcess(state_mod_corr_noise_test, T, state_dim)

        if self.force_noise_train is not None:
            force_mod_corr_noise_train = self.force_noise_train["mod_corr_noise"]
            self.train_force_noise_vec = VARMAProcess(force_mod_corr_noise_train, T, force_dim)
        if self.force_noise_test is not None:
            force_mod_corr_noise_test = self.force_noise_test["mod_corr_noise"]
            self.test_force_noise_vec = VARMAProcess(force_mod_corr_noise_test, T, force_dim)

        self.dist_parameters_train()

        self.step_counter = 0
        obs = super().reset()  # NOTE: We are currently NOT noising the initial env state!

        return obs

    def set_noise(self, kwargs):
        if "obs_noise_train" in kwargs:
            self.obs_noise_train = kwargs["obs_noise_train"]
        if "state_noise_train" in kwargs:
            self.state_noise_train = kwargs["state_noise_train"]
        if "force_noise_train" in kwargs:
            self.force_noise_train = kwargs["force_noise_train"]
        if "force_vec_dict_train" in kwargs:
            self.force_vec_dict_train = kwargs["force_vec_dict_train"]
        if "obs_noise_test" in kwargs:
            self.obs_noise_test = kwargs["obs_noise_test"]
        if "state_noise_test" in kwargs:
            self.state_noise_test = kwargs["state_noise_test"]
        if "force_noise_test" in kwargs:
            self.force_noise_test = kwargs["force_noise_test"]
        if "force_vec_dict_test" in kwargs:
            self.force_vec_dict_test = kwargs["force_vec_dict_test"]
        if "injection_time" in kwargs:
            self.injection_time = kwargs["injection_time"]
        pass
