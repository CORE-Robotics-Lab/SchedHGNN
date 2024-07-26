import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import PIL.Image as Image
import gym
import random
import math
from scipy.stats import multivariate_normal
import curses
from gym import spaces
import time
import os
from waves import WaveData
from .multiagentenv import MultiAgentEnv
from gym.spaces import Discrete
import re
import copy


class Marine(gym.Env):
    def __init__(self, args,
                        n_enemies=0,
                        moving_logis=True,
                        no_stay=False,
                        enemy_comm=True,
                        second_reward_scheme=False,
                        ):

        self.__version__ = "0.0.1"
        self.args = args
        '''defaulty params'''
        self.tensor_obs = args.tensor_obs
        self.OUTSIDE_CLASS = 1
        self.LOGISTICS_CLASS = 2
        self.ROUTING_CLASS = 3
        self.FUEL_START_IDX = 4
        self.TIMESTEP_PENALTY = -0.05
        self.LOGIS_REWARD = 0
        self.POS_LOGIS_REWARD = 0.05
        self.POS_DEST_REWARD = 0.05
        self.CONNECT_REWARD = 0.05
        self.episode_over = False
        self.moving_logis = moving_logis
        '''param arguments'''
        self.nfriendly_P = args.num_P # num of routing agents
        self.nfriendly_A = args.num_A # num of logistics agents
        self.vision = args.vision
        self.intensity_levels = args.intensity_levels
        self.num_dest = args.num_dest
        if not args.tensor_obs and self.num_dest != 1:
            self.num_dest = 1
            print('Caution - currently only 1 destination case is supported when tensor_obs=False, args.num_dest is changed from %d to %d'%(args.num_dest, 1))
            args.num_dest = 1

        self.limited_refuel = args.limited_refuel
        ### no enemies in ocean env
        self.nrouting = self.nfriendly_P
        self.nlogis = self.nfriendly_A
        dim = args.dim
        self.dims = (dim, dim)
        self.dim = dim
        self.mode = args.mode # default cooperative
        self.enemy_comm = enemy_comm
        self.stay = not no_stay
        self.shared_reward = False # always individual reward
        self.second_reward_scheme = second_reward_scheme
        self.episode_limit = args.episode_limit
        self.n_agents = self.nfriendly_P
        self.n_total = self.nrouting
        if self.enemy_comm:
            self.n_total += self.nlogis
        self._episode_steps = 0
        self.battles_won = 0
        self.battles_game = 0
        self.timeouts = 0
        self.total_steps = 0

        if self.stay:
            self.naction = 5
        else:
            self.naction = 4
        self.BASE = (self.dims[0] * self.dims[1])
        self.binary_BASE = self.BASE.bit_length() # binarized value
        self.OUTSIDE_CLASS += 2*self.binary_BASE
        self.LOGISTICS_CLASS += 2*self.binary_BASE
        self.ROUTING_CLASS += 2*self.binary_BASE
        self.FUEL_START_IDX += 2*self.binary_BASE
        self.state_len = (self.nfriendly_P + self.nfriendly_A + 1 + self.num_dest)  # Change this if you change state_len


        # Maximum fuel routing agent can take at once
        self.max_fuel = int(self.dim)
        self.binary_max_fuel = self.max_fuel.bit_length()
        self.max_fuel_logis = self.BASE  # dim * dim
        if not self.limited_refuel:
            self.fuel_left = np.array([self.max_fuel] * self.nrouting)
        else:  # self.limited_refuel
            self.fuel_left = np.array([self.max_fuel] * (self.nrouting + self.nlogis))
            self.fuel_left[self.nrouting:] = self.max_fuel_logis

        self.vocab_size = self.binary_BASE + self.binary_BASE + self.binary_max_fuel + (self.n_total + 1 + 1 + 2) * (2*self.vision+1)**2
        #         grid (for agent's loc and dest_pos) + fuel level (0~max_fuel) + (num_agents + outside + dest + intensity levels (4 levels - 2 bits)) of observable range
        print('environment self.vocab_size: ', self.vocab_size)

        if self.tensor_obs:
            self.observation_space_tens = spaces.Box(-np.inf, np.inf, shape=[self.dim, self.dim, 4])

            self.feature_map = [np.zeros((4, 2*self.vision+1, 2*self.vision+1)), np.zeros((self.dim, self.dim, 1))]  # Channels: 1, 2, 3 each for logis, routing, dest & 4 for pairing from scheduler & 5 for intensity
            self.scheduling_feature_map = [np.zeros((5, 2 * self.vision + 1, 2 * self.vision + 1)), np.zeros((self.dim, self.dim, 1))]  # Channels: 1, 2, 3 each for logis, routing, dest & 4 for pairing from scheduler & 5 for intensity


            self.true_feature_map = np.zeros((self.dim, self.dim, 4 + self.nfriendly_P + self.nfriendly_A))
            self.true_feature_scheduling_map = np.zeros((self.dim, self.dim, 5 + self.nfriendly_P + self.nfriendly_A))
        else:
            self.observation_space = spaces.Box(low=0, high=1,
                                                shape=(self.vocab_size, (2 * self.vision) + 1, (2 * self.vision) + 1),
                                                dtype=int)
            # Actual observation will be of the shape 1 * npredator * (2v+1) * (2v+1) * vocab_size

        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        for i in range(self.n_total):
            self.action_space.append(Discrete(self.naction))
            self.share_observation_space.append(self.get_state_size())

            if self.tensor_obs and not self.args.binarized_obs_comm:
                self.observation_space.append([2])
            else:
                self.observation_space.append(self.get_obs_size())

        self.num_block_y = self.dim
        self.num_block_x = self.dim
        self.WaveData = WaveData()
        self.whole_intensity_data = self.WaveData.get_region_intensity_all_data()  # spatio-temporal data

        self.intensity_data = self.whole_intensity_data
        self.current_step = 0
        self.current_epoch_rate = -1

        print('MarineMultiEnv is initialized')

    def get_stats(self):
        stats = {
            "battles_won": self.battles_won,
            "battles_game": self.battles_game,
            "battles_draw": self.timeouts,
            "win_rate": self.battles_won / self.battles_game,
            "timeouts": self.timeouts,
            "steps_taken": self.total_steps / self.battles_game,  # average steps taken
            "agent_locs": self.agentlocs
        }
        return stats


    def render(self):
        '''not implemented'''
        pass
        raise NotImplementedError


    def close(self):
        '''not applicable to PP or PCP'''
        pass
        raise NotImplementedError


    def seed(self, seed):
        self._seed = seed
        return


    def save_replay(self):
        '''not implemented'''
        pass
        raise NotImplementedError

    def pick_intensity_data(self):

        if hasattr(self,'dims'): # randomly sample it
            start_lat = np.random.choice(self.whole_intensity_data.shape[0] - self.dims[0])
            start_long = np.random.choice(self.whole_intensity_data.shape[1] - self.dims[1])
            start_time = np.random.choice(self.whole_intensity_data.shape[2] - self.episode_limit)
            self.intensity_data = self.whole_intensity_data[start_lat:start_lat + self.dims[0],
                                  start_long:start_long + self.dims[1], start_time:start_time + self.episode_limit]

        else: # the first episode
            start_lat = 0
            start_long = 0
            start_time = 0
            if hasattr(self, 'dims'):
                self.intensity_data = self.whole_intensity_data[start_lat:start_lat + self.dims[0],
                                      start_long:start_long + self.dims[1], start_time:start_time + self.episode_limit]
            else:
                self.intensity_data = self.whole_intensity_data

    def step(self, actions):
        """
        The agents take a step in the environment.
        Parameters
        ----------
        action : list/ndarray of length m, containing the indexes of what lever each 'm' chosen agents pulled.
        Returns
        -------
        obs, reward, episode_over, info : tuple
            obs (object) :
            reward (float) : Ratio of Number of discrete levers pulled to total number of levers.
            episode_over (bool) : Will be true as episode length is 1
            info (dict) : diagnostic information useful for debugging.
        """
        self.intensity_update()
        # assert not self.episode_over
        if self.episode_over:
            raise RuntimeError("Episode is done")

        action = actions
        action = np.atleast_1d(action)
        infos = [{} for i in range(self.n_total)]  # info per num. of routing agents
        # Dones -- take care of only routing agents
        dones = np.zeros((self.n_total), dtype=bool)
        bad_transition = False
        if len(action) == 1:
            action = action[0]
        for i, a in enumerate(action):
            self._take_action(i, a)
        self._episode_steps += 1
        self.total_steps += 1
        assert np.all(action <= self.naction), "Actions should be in the range [0,naction)."


        if self.tensor_obs and not self.args.binarized_obs_comm:
            self.obs = self._get_tensor_obs()
        else:
            self.obs = self._get_obs()
        self.state = self.get_state()

        if self._episode_steps >= self.episode_limit:
            self.episode_over = True
            self.battles_game += 1
            self.timeouts += 1
            bad_transition = True
        reward = self._get_reward()

        assert self.current_step < self.episode_limit
        self.current_step += 1


        for i in range(self.n_total):
            infos[i] = {"battles_won": self.battles_won,
                        "battles_game": self.battles_game,
                        "battles_draw": self.timeouts,
                        "bad_transition": bad_transition,
                        "won": self.battle_won,
                        "episode_steps": self._episode_steps,
                        "locs": np.vstack([self.routing_loc, self.logis_loc]),
                        "dest": self.dest_pos,
                        "dest_id": self.agentwise_dest_id,
                        "fuel_left": self.fuel_left
                        }

            if self.episode_over:
                dones[i] = True

        return self.obs, self.state, reward, dones, infos, self.get_avail_actions()

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.current_epoch_rate = -1
        # Reset the fuel consumed
        if not self.limited_refuel:
            self.fuel_left = np.array([self.max_fuel] * self.nrouting)
        else: #self.limited_refuel
            self.fuel_left = np.array([self.max_fuel] * (self.nrouting+self.nlogis))
            self.fuel_left[self.nrouting:] = self.max_fuel_logis



        self.pick_intensity_data()
        self.current_step = 0
        self.intensity_update()

        self.episode_over = False
        self.success = False
        self.reached_dest = np.zeros(self.nrouting)
        self._episode_steps = 0
        self.battle_won = False


        self.dest_pos = [[np.random.randint(self.num_block_y), self.num_block_x - 1] for _ in range(self.num_dest)]
        self.agentwise_dest_id = np.random.randint(self.num_dest, size=self.nrouting)
        self.dest_pos_encode = [x[0] * self.dims[0] + x[1] for x in self.dest_pos]


        # Initial location of routing and logistics agents
        locs = self._get_coordinates()
        self.routing_loc, self.logis_loc = locs[:self.nrouting], locs[self.nrouting:]
        self._set_grid()


        # stat - like success ratio
        self.stat = dict()


        # Observation will be nrouting * vision * vision ndarray ## WHY???
        if self.tensor_obs and not self.args.binarized_obs_comm:
            self.obs = self._get_tensor_obs()
        else:
            self.obs = self._get_obs()

        self.state = self.get_state()
        return self.obs, self.state, self.get_avail_actions()

    def get_binarized_seq(self, decimal, num_of_bits):
        # decimal = pos[0] * self.dims[0] + pos[1]
        binary = bin(decimal)
        seq = binary.lstrip('0b')
        seq = seq.zfill(num_of_bits)
        ones_locs = [m.start() for m in re.finditer('1', seq)]

        return np.array(ones_locs)

    def get_info(self):
        infos = [{} for i in range(self.n_total)]  # info per num. of routing agents
        for i in range(self.n_total):
            infos[i] = {"battles_won": self.battles_won,
                        "battles_game": self.battles_game,
                        "battles_draw": self.timeouts,
                        "bad_transition": False,
                        "won": self.battle_won,
                        "episode_steps": self._episode_steps,
                        "locs": np.vstack([self.routing_loc, self.logis_loc]),
                        "dest": self.dest_pos,
                        "dest_id": self.agentwise_dest_id,
                        "fuel_left": self.fuel_left
                        }
        return infos

    def curr_epoch_update(self, epoch_rate):
        self.current_epoch_rate = epoch_rate

    def intensity_update(self):
        self.intensity = self.intensity_data[:, :, self.current_step]


    def sample_init_position_of_logistics(self):
        if self.args.logis_init_position_over_the_space:
            idx = np.random.choice(np.prod(self.dims), (self.nlogis), replace=False)
            list_of_locs = np.vstack(np.unravel_index(idx, self.dims)).T
            return list(list_of_locs[:, 1]), list(list_of_locs[:, 0])
        else:
            raise NotImplementedError



    def _get_coordinates(self):

        x_init_routing = [0] * self.nrouting  # horizontally left edge
        y_init_routing = list(np.random.choice(self.num_block_y, self.nrouting, replace=False))  # random sampling
        x_logis, y_logis = self.sample_init_position_of_logistics()
        x = x_init_routing + x_logis
        y = y_init_routing + y_logis
        return np.vstack((np.array(y), np.array(x))).T


    def _set_grid(self):
        self.grid = np.arange(self.BASE).reshape(self.dims)

        # Padding for vision
        self.grid = np.pad(self.grid, self.vision, 'constant', constant_values=-1)


    def _get_obs(self):
        # self.vocab_size = self.binary_BASE + self.binary_BASE + self.binary_max_fuel + (self.n_total + 1 + 1 + 2) * (2 * self.vision + 1) ** 2
        #         grid (for agent's loc and dest_pos) + fuel level (0~max_fuel) + (num_agents + outside + dest + intensity levels (4 levels - 2 bits)) of observable range

        bool_base_grid = np.zeros((self.dims[0] + 2*self.vision, self.dims[1] + 2*self.vision, self.n_total + 1 + 1 + 2))
        # axis=2 0~self.ntotal-1: existing agents' ids at the grid
        # axis=2 self.ntotal: either outside or not
        # axis=2 self.ntotal+1: either dest or not
        # axis=2 self.ntotal+2~self.ntotal+3: intensity

        # Outside notation
        bool_base_grid[0, :, self.n_total] = 1
        bool_base_grid[-1, :, self.n_total] = 1
        bool_base_grid[:, 0, self.n_total] = 1
        bool_base_grid[:, -1, self.n_total] = 1

        # Dest encoding
        for dst in self.dest_pos:
            bool_base_grid[dst[0] + self.vision, dst[1] + self.vision, self.n_total+1] = 1

        # Wavemap encoding
        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                if self.intensity[i, j] == 0:  # 00
                    pass
                elif self.intensity[i, j] == 1 / 3:  # 01
                    bool_base_grid[i + self.vision, j + self.vision, -1] = 1
                elif self.intensity[i, j] == 2 / 3:  # 10
                    bool_base_grid[i + self.vision, j + self.vision, -2] = 1
                else:  # self.intensity[i, j] == 1 # 11
                    bool_base_grid[i + self.vision, j + self.vision, -1] = 1
                    bool_base_grid[i + self.vision, j + self.vision, -2] = 1

        # agent ids
        all_locs = np.vstack((self.routing_loc, self.logis_loc))
        for i, loc in enumerate(all_locs):
            bool_base_grid[loc[0]+self.vision, loc[1]+self.vision, i] = 1

        obs = []
        for i, loc in enumerate(all_locs):
            agent_pos_dst_fuel = np.zeros(2 * self.binary_BASE + self.binary_max_fuel)
            binary_pos = self.get_binarized_seq(loc[0]*self.dim+loc[1], self.binary_BASE)
            binary_dst = self.binary_BASE + self.get_binarized_seq(self.dest_pos_encode[0], self.binary_BASE) # only one destination case is supported as of now

            if len(binary_pos) > 0:
                agent_pos_dst_fuel[binary_pos] = 1
            if len(binary_dst) > 0:
                agent_pos_dst_fuel[binary_dst] = 1
            if i < self.nrouting:
                binary_fuel = self.get_binarized_seq(self.fuel_left[i], self.binary_max_fuel)
                if len(binary_fuel) > 0:
                    binary_fuel = binary_fuel + 2 * self.binary_BASE
                    agent_pos_dst_fuel[binary_fuel] = 1
            elif self.limited_refuel: # logis agents only have fuel_left when limited_refuel
                binary_fuel = self.get_binarized_seq(int(self.fuel_left[i] / self.max_fuel), self.binary_max_fuel)
                if len(binary_fuel) > 0:
                    binary_fuel = binary_fuel + 2 * self.binary_BASE
                    agent_pos_dst_fuel[binary_fuel] = 1

            slice_y = slice(loc[0], loc[0] + (2 * self.vision) + 1)
            slice_x = slice(loc[1], loc[1] + (2 * self.vision) + 1)
            agent_around_info = bool_base_grid[slice_y, slice_x]

            agent_obs = np.concatenate((agent_pos_dst_fuel, agent_around_info.reshape(-1)))
            obs.append(agent_obs)

        return obs


    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self._get_obs()[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """

        return [self.vocab_size]


    def get_state(self):

        state = np.vstack((self.routing_loc, self.logis_loc, self.dest_pos))

        state = np.reshape(state, (-1))  # flatten
        state = np.concatenate([state, self.fuel_left])

        state = np.append(state, self._episode_steps / self.episode_limit)
        self.state = []
        for i in range(self.n_total):
            self.state.append(state)
        return self.state

    def get_state_size(self):
        """ Returns the shape of the state"""

        state_size = self.n_total * 2 + self.num_dest * 2 + 1
        state_size += self.nrouting
        if self.limited_refuel:
            state_size += self.nlogis

        return [state_size]


    def get_avail_actions(self):
        ''' all actions are available'''
        avail_actions = []
        for i in range(self.n_total):
            avail_actions.append([1] * self.naction)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        '''should be all the same'''
        return self.get_avail_actions()[agent_id]

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return self.naction

    def _get_tensor_obs(self):
        """
        Notes: axis is initialized from the top corner. For example, (1,0) (one down, 0 right) refers to
        array([[0., 0., 0., 0., 0.],
               [1., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0.]])
        """
        # this code is based on the assumption that self.vision = 1
        assert self.vision == 1
        ### in the above, self.true_feature_map = np.zeros((self.dim, self.dim, 4 + self.nfriendly_P + self.nfriendly_A))
        true_feature_map = self.true_feature_map.copy()

        for i in ['routing', 'logistics', 'self_fuel', 'self']:
            if i == 'routing':
                for j, p in enumerate(self.routing_loc):
                    true_feature_map[p[0], p[1], 0] += 1
            elif i == 'logistics':
                for j, p in enumerate(self.logis_loc):
                    true_feature_map[p[0], p[1], 1] += 1
            elif i == 'self_fuel':
                pass # later fuel level will be recorded
            elif i == 'self':
                all_locs = np.vstack([self.routing_loc, self.logis_loc])
                for j, p in enumerate(all_locs):
                    true_feature_map[p[0], p[1], 4 + j] = j+1

                    if j < self.nfriendly_P:  # routing
                        agent_dest = self.dest_pos[self.agentwise_dest_id[j]]
                        true_feature_map[agent_dest[0], agent_dest[1], 4 + j] = -1  # to encode its dest_pos
                    else:  # j >= self.nfriendly_P, logis
                        for k in self.dest_pos:
                            true_feature_map[k[0], k[1], 4 + j] = -1  # logis knows all dest

        true_feature_map[:, :, 3] = self.intensity[:, :] # it encodes wave map

        obs = []
        agent_counter = 0

        for j, p in enumerate(self.routing_loc):

            sub_obs = copy.deepcopy(self.feature_map)


            # self.vision is always larger than 0
            y_start = p[0]  # + 1 - self.vision
            y_end = p[0] + 2 * self.vision + 1  # + self.vision + 1 + 1
            x_start = p[1]  # + 1 - self.vision
            x_end = p[1] + 2 * self.vision + 1  # 1 + self.vision + 1
            padded_feature_map_0 = np.pad(true_feature_map[:, :, 0], self.vision, constant_values=-1)
            padded_feature_map_1 = np.pad(true_feature_map[:, :, 1], self.vision, constant_values=-1)
            padded_feature_map_2 = np.pad(true_feature_map[:, :, 2], self.vision, constant_values=-1)
            padded_feature_map_3 = np.pad(true_feature_map[:, :, 3], self.vision, constant_values=-1)


            sub_obs[0][0] = padded_feature_map_0[y_start:y_end, x_start:x_end]
            sub_obs[0][1] = padded_feature_map_1[y_start:y_end, x_start:x_end]
            sub_obs[0][2] = padded_feature_map_2[y_start:y_end, x_start:x_end]
            sub_obs[0][3] = padded_feature_map_3[y_start:y_end, x_start:x_end]
            sub_obs[1] = true_feature_map[:, :, 4 + agent_counter]

            # Record fuel level
            sub_obs[0][2][self.vision, self.vision] = self.fuel_left[j]

            agent_counter += 1
            obs.append(sub_obs)

        for j, p in enumerate(self.logis_loc):

            sub_obs = copy.deepcopy(self.feature_map)
            # self.vision is always larger than 0
            y_start = p[0]  # + 1 - self.vision
            y_end = p[0] + 2 * self.vision + 1  # + self.vision + 1 + 1
            x_start = p[1]  # + 1 - self.vision
            x_end = p[1] + 2 * self.vision + 1  # 1 + self.vision + 1
            padded_feature_map_0 = np.pad(true_feature_map[:, :, 0], self.vision, constant_values=-1)
            padded_feature_map_1 = np.pad(true_feature_map[:, :, 1], self.vision, constant_values=-1)
            padded_feature_map_2 = np.pad(true_feature_map[:, :, 2], self.vision, constant_values=-1)
            padded_feature_map_3 = np.pad(true_feature_map[:, :, 3], self.vision, constant_values=-1)

            sub_obs[0][0] = padded_feature_map_0[y_start:y_end, x_start:x_end]
            sub_obs[0][1] = padded_feature_map_1[y_start:y_end, x_start:x_end]
            sub_obs[0][2] = padded_feature_map_2[y_start:y_end, x_start:x_end]
            sub_obs[0][3] = padded_feature_map_3[y_start:y_end, x_start:x_end]
            sub_obs[1] = true_feature_map[:, :, 4 + agent_counter]

            # Record fuel level
            if self.limited_refuel:
                sub_obs[0][2][self.vision, self.vision] = self.fuel_left[j+self.nrouting]
            else:
                sub_obs[0][2][self.vision, self.vision] = self.dim * self.dim  # to denote this logis agent has enough fuel

            agent_counter += 1
            obs.append(sub_obs)


        obs = np.stack(obs)

        return obs

    def _take_action(self, idx, act):
        # routing agents action
        if idx >= self.nrouting:
            # fixed logistics agents
            if not self.moving_logis:
                return
            # moving logistics agents
            else:
                if act == 4:
                    return
                # UP
                if act == 0 and self.grid[max(0,
                                              self.logis_loc[idx - self.nrouting][0] + self.vision - 1),
                                          self.logis_loc[idx - self.nrouting][1] + self.vision] != -1:
                    self.logis_loc[idx - self.nrouting][0] = max(0, self.logis_loc[idx - self.nrouting][0] - 1)

                # RIGHT
                elif act == 1 and self.grid[self.logis_loc[idx - self.nrouting][0] + self.vision,
                                            min(self.dims[1] - 1,
                                                self.logis_loc[idx - self.nrouting][1] + self.vision + 1)] != -1:
                    self.logis_loc[idx - self.nrouting][1] = min(self.dims[1] - 1,
                                                   self.logis_loc[idx - self.nrouting][1] + 1)

                # DOWN
                elif act == 2 and self.grid[min(self.dims[0] - 1,
                                                self.logis_loc[idx - self.nrouting][0] + self.vision + 1),
                                            self.logis_loc[idx - self.nrouting][1] + self.vision] != -1:
                    self.logis_loc[idx - self.nrouting][0] = min(self.dims[0] - 1,
                                                   self.logis_loc[idx - self.nrouting][0] + 1)

                # LEFT
                elif act == 3 and self.grid[self.logis_loc[idx - self.nrouting][0] + self.vision,
                                            max(0,
                                                self.logis_loc[idx - self.nrouting][1] + self.vision - 1)] != -1:
                    self.logis_loc[idx - self.nrouting][1] = max(0, self.logis_loc[idx - self.nrouting][1] - 1)

        else:
            if self.reached_dest[idx] == 1:
                return

            # STAY action
            if act==4:
                return

            # UP
            if act==0 and self.grid[max(0,
                                    self.routing_loc[idx][0] + self.vision - 1),
                                    self.routing_loc[idx][1] + self.vision] != -1:
                self.routing_loc[idx][0] = max(0, self.routing_loc[idx][0]-1)

            # RIGHT
            elif act==1 and self.grid[self.routing_loc[idx][0] + self.vision,
                                    min(self.dims[1] -1,
                                        self.routing_loc[idx][1] + self.vision + 1)] != -1:
                self.routing_loc[idx][1] = min(self.dims[1]-1,
                                                self.routing_loc[idx][1]+1)

            # DOWN
            elif act==2 and self.grid[min(self.dims[0]-1,
                                        self.routing_loc[idx][0] + self.vision + 1),
                                        self.routing_loc[idx][1] + self.vision] != -1:
                self.routing_loc[idx][0] = min(self.dims[0]-1,
                                                self.routing_loc[idx][0]+1)

            # LEFT
            elif act==3 and self.grid[self.routing_loc[idx][0] + self.vision,
                                        max(0,
                                        self.routing_loc[idx][1] + self.vision - 1)] != -1:
                self.routing_loc[idx][1] = max(0, self.routing_loc[idx][1]-1)


    def get_taxi_distance(self, pos_x, pos_y):
        return abs(pos_x[0] - pos_y[0]) + abs(pos_x[1] - pos_y[1])

    def reward_terminal(self):
        return np.zeros_like(self._get_reward())


    def _all_idx(self, idx, axis):
        grid = np.ogrid[tuple(map(slice, idx.shape))]
        grid.insert(axis, idx)
        return tuple(grid)


    def _get_reward(self):

        n = self.nrouting if (not self.moving_logis) else self.nrouting + self.nlogis

        reward = np.full(n, -0.05) # each time step, agents get timestep penalty to not slack off
        on_dest = []
        not_on_dest = []
        just_arrived_dest = []
        for j in range(self.nrouting):
            if np.all(self.routing_loc[j] == self.dest_pos[self.agentwise_dest_id[j]]):
                on_dest.append(j)
                if self.reached_dest[j] != 1:
                    self.reached_dest[j] = 1
                    just_arrived_dest.append(j)
            else:
                not_on_dest.append(j)
                dist_to_dest = self.get_taxi_distance(self.routing_loc[j], self.dest_pos[self.agentwise_dest_id[j]])
                reward[j] = -0.05 * dist_to_dest # more penalty for agents further from the destination

        on_dest = np.array(on_dest, dtype=int)
        not_on_dest = np.array(not_on_dest, dtype=int)
        just_arrived_dest = np.array(just_arrived_dest, dtype=int)
        nb_routing_on_dest = on_dest.size
        nb_just_arrived = just_arrived_dest.size


        if self.mode == 'cooperative':
            # agents on destination are rewarded every time an agent is arriving
            reward[on_dest] = 0.1 * nb_just_arrived
        else:
            raise RuntimeError("Incorrect mode, Available modes: [cooperative]")


        if np.all(self.reached_dest == 1) and ((self.mode == 'mixed') or (self.mode == 'cooperative')):
            # self.episode_over = True
            # self.success = True
            self.battles_won += 1
            self.battles_game += 1
            self.episode_over = True
            self.battle_won = True


        # Success ratio
        if self.mode != 'competitive':
            if nb_routing_on_dest == self.nrouting:
                self.stat['success'] = 1

            else:
                self.stat['success'] = 0

        ## If routing agents meet an logistics agent (refuelling), logistics agents get reward
        logis_loc_connected = {}
        for idx_l in range(self.nlogis):
            if self.limited_refuel:
                assert self.fuel_left[self.nrouting + idx_l] >= 0
                assert self.fuel_left[self.nrouting + idx_l] <= self.max_fuel_logis
            logis_loc_connected[idx_l] = []
        for idx_r, p in enumerate(self.routing_loc):

            assert self.fuel_left[idx_r] >= 0
            assert self.fuel_left[idx_r] <= self.max_fuel
            # If this agent has arrived the destination, do not do anything since we already gave them the reward
            if idx_r in on_dest:
                continue
            # If fuel_left=0, get minus rewards for both routing & logistic agents
            if self.fuel_left[idx_r] == 0:
                reward[:] = -10.0
                self.episode_over = True

                break

            for idx_l, q in enumerate(self.logis_loc):
                if p[0] == q[0] and p[1] == q[1]:
                    # If connected, logistic agent refuel the routing agent according to its intensity
                    logis_loc_connected[idx_l].append(idx_r)

                    # Logistic agents get reward
                    if self.moving_logis and ((not self.limited_refuel) or (self.fuel_left[self.nrouting + idx_l] > 0)):
                        logis_intensity = self.intensity[p[0], p[1]]
                        reward[self.nrouting + idx_l] = self.CONNECT_REWARD * logis_intensity

            self.fuel_left[idx_r] -= 1  # Each time, routing agents not at the dest consume fuel by the amount of 1

        # Refuel
        if not self.episode_over:
            for idx_l in range(self.nlogis):
                logis_pos = self.logis_loc[idx_l]  # logistics agent's position
                logis_intensity = self.intensity[logis_pos[0], logis_pos[1]]  # intensity level
                logis_connected = logis_loc_connected[idx_l]  # connected routing agents' indexes

                if self.dim == 5:  # (0,1,2,3) for each intensity level
                    logis_index = int(logis_intensity * 3)
                    refuel_levels_per_intensity = [0, 1, 2, 3]
                    refuel_per_timestep = refuel_levels_per_intensity[logis_index]
                elif self.dim == 10:  # (0,1,3,5) for each intensity level
                    logis_index = int(logis_intensity * 3)
                    refuel_levels_per_intensity = [0, 1, 3, 5]
                    refuel_per_timestep = refuel_levels_per_intensity[logis_index]
                elif self.dim == 15:  # (0,3,5,8) for each intensity level
                    logis_index = int(logis_intensity * 3)
                    refuel_levels_per_intensity = [0, 3, 5, 8]
                    refuel_per_timestep = refuel_levels_per_intensity[logis_index]
                elif self.dim == 20:  # (0,4,7,10) for each intensity level
                    logis_index = int(logis_intensity * 3)
                    refuel_levels_per_intensity = [0, 4, 7, 10]
                    refuel_per_timestep = refuel_levels_per_intensity[logis_index]
                else:
                    refuel_per_timestep = int((self.max_fuel / 2) * logis_intensity)

                fuel_connected = {}
                for i in logis_connected:
                    fuel_connected[i] = self.fuel_left[i]

                logis_connected.sort(
                    key=lambda x: fuel_connected[x])  # sort the routing agents' order by fuel levels, ascending
                if self.limited_refuel and len(logis_connected) > 2:  # a logistics agent can refuel up to 2 routing agents: refuel routing agents with lower fuel levels
                    logis_connected = logis_connected[:2]

                if not self.limited_refuel:
                    for idx_r in logis_connected:
                        self.fuel_left[idx_r] += refuel_per_timestep
                        if self.fuel_left[idx_r] > self.max_fuel:  # clamp the amount of fuel
                            self.fuel_left[idx_r] = self.max_fuel
                else:  # self.limited_refuel
                    for idx_r in logis_connected:
                        max_refuel_amount = min(self.max_fuel - self.fuel_left[idx_r], refuel_per_timestep)
                        refuel_amount = min(max_refuel_amount, self.fuel_left[self.nrouting + idx_l])
                        self.fuel_left[idx_r] += refuel_amount
                        self.fuel_left[self.nrouting + idx_l] -= refuel_amount
                        # if self.fuel_left[idx_r] > self.max_fuel:  # clamp the amount of fuel
                        #     self.fuel_left[idx_r] = self.max_fuel

        return reward
