#!/usr/bin/env python
import json
import time
import wandb
import copy
import numpy as np
from functools import reduce
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from collections import defaultdict, deque
from onpolicy.runner.shared.base_runner import Runner
import torch.nn as nn

import cv2
from PIL import Image

def _t2n(x):
    return x.detach().cpu().numpy()


class MarineRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for Marine environment. See parent class for details."""

    def __init__(self, config):
        super(MarineRunner, self).__init__(config)
        print("Init runner")

    def run(self):
        print("runnning")
        self.warmup()

        start = time.time()

        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)
        train_episode_length = []
        train_battles_won = 0
        train_battles_game = 0

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            # print("in episode")
            episode_start = time.time()
            if self.all_args.schedule_rewards:
                infos = self.envs.get_info()
                # print('infos', infos)
                position = infos[0]["locs"]
                dest = infos[0]["dest"]
                dest_id = infos[0]["dest_id"]
                fuel_left = infos[0]["fuel_left"]

            for step in range(self.episode_length):
                time_init = time.time()

                if self.all_args.schedule_rewards:
                    # schedule is generated from agents' positions and fuel levels.
                    # If schedule[i] < self.num_agents: it means ith agent is paired with another agent and go toward that agent's position
                    # If schedule[i] == self.num_agents: it means ith agent should go to its desired destination directly
                    # An agent will take action sampled from policy & get reward when it accomplish the previous schedule

                    pairing = self.get_schedules(position, dest, dest_id, fuel_left)

                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_obs, rnn_states_critic = self.collect(step)

                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)
                rewards = rewards.reshape((1, -1, 1))
                if self.all_args.schedule_rewards:
                    new_position = infos[0][0]["locs"]

                    schedule_pairing_rewards = self.get_pairing_reward_from_scheduling(pairing, position, new_position, dest, dest_id)
                    schedule_pairing_rewards = schedule_pairing_rewards.reshape(rewards.shape)
                    total_rewards = rewards + schedule_pairing_rewards

                    position = new_position
                    fuel_left = infos[0][0]["fuel_left"]

                for e in range(self.n_rollout_threads):
                    if np.all(dones[e]):
                        train_episode_length.append(infos[e][0]['episode_steps'])
                        train_battles_game += 1
                        if infos[e][0]['won']:
                            train_battles_won += 1


                # For every global step, update the full and local maps
                if not self.all_args.schedule_rewards:
                    data = obs, share_obs, rewards, dones, infos, available_actions, \
                           values, actions, action_log_probs, \
                           rnn_states, rnn_obs, rnn_states_critic
                else:
                    data = obs, share_obs, total_rewards, dones, infos, available_actions, \
                           values, actions, action_log_probs, \
                           rnn_states, rnn_obs, rnn_states_critic

                # insert data into buffer
                self.insert(data)

                # print('Total time stepping through episode {}'.format(time.time() - time_init))

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # monitor episode wall-click time
            episode_end = time.time()
            episode_time = episode_end - episode_start
            # print("\t >>> episode_time: " + str(episode_time))

            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save(episode // 10)



            # save logs
            if episode % self.log_interval == 0 or episode == episodes - 1:
                self.logger.save_stats()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                total_time = (end - start) / 3600  # convert to hours
                print("\n Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n".format(
                    self.algorithm_name,
                    self.experiment_name,
                    episode,
                    episodes,
                    total_num_steps,
                    self.num_env_steps,
                    int(total_num_steps / (end - start))))

                if 'Marine' in self.env_name:
                    train_episode_length_mean = np.mean(np.array(train_episode_length))

                    train_win_rate = train_battles_won / train_battles_game
                    train_infos['training_ep_length_mean'] = train_episode_length_mean
                    train_infos['training_win_rate'] = train_win_rate
                    print(f"training episode length is {train_episode_length_mean}")
                    print(f"training win rate is {train_win_rate}")

                    train_battles_game = 0
                    train_battles_won = 0
                    train_episode_length = []
                    battles_won = []
                    battles_game = []
                    incre_battles_won = []
                    incre_battles_game = []
                    # success = []

                    for i, info in enumerate(infos):
                        # print('Debug', i, info)
                        if 'battles_won' in info[0].keys():
                            battles_won.append(info[0]['battles_won'])
                            incre_battles_won.append(info[0]['battles_won'] - last_battles_won[i])
                        if 'battles_game' in info[0].keys():
                            battles_game.append(info[0]['battles_game'])
                            incre_battles_game.append(info[0]['battles_game'] - last_battles_game[i])


                    last_battles_game = battles_game
                    last_battles_won = battles_won

                self.log_train(train_infos, total_num_steps, total_time)

            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps, total_time)

    def warmup(self):

        # reset env
        obs, share_obs, available_actions = self.envs.reset()

        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def get_schedules(self, position, dest, dest_id, fuel_left):
        routing_agents_that_need_pairing = []
        distance_to_dest_of_routing_agents_that_need_pairing = []
        logistics_agents_that_can_be_paired = np.array(list(range(self.all_args.num_P, self.num_agents)))

        for i in range(self.all_args.num_P):
            agent_destination = dest[dest_id[i]]
            agent_position = position[i]
            distance_to_dest = self.get_taxi_distance(agent_destination, agent_position)
            agent_fuel = fuel_left[i]
            distance_to_dest_of_routing_agents_that_need_pairing.append(distance_to_dest)

            if distance_to_dest > agent_fuel:
                routing_agents_that_need_pairing.append(i)

        routing_agents_that_need_pairing.sort(key=lambda x: fuel_left[x],
                                              reverse=True)  # descending, that means, the routing agents further from its destination will be paired first


        pairing = np.full((self.num_agents), -1)

        for r in routing_agents_that_need_pairing:
            agent_position = position[r]
            if len(logistics_agents_that_can_be_paired) == 0:
                break
            distance_to_remaining_logis = []
            for l in logistics_agents_that_can_be_paired:
                logis_position = position[l]
                distance_to_logis = self.get_taxi_distance(agent_position, logis_position)
                distance_to_remaining_logis.append(distance_to_logis)

            sorted_index = np.argsort(distance_to_remaining_logis) # ascending
            logistics_agents_that_can_be_paired = logistics_agents_that_can_be_paired[sorted_index] # ascending, that means, the logis agent that is closest to the routing agent will be paired

            paired_logis = logistics_agents_that_can_be_paired[0]
            pairing[r] = paired_logis
            pairing[paired_logis] = r
            logistics_agents_that_can_be_paired = logistics_agents_that_can_be_paired[1:]

        return pairing

    @torch.no_grad()
    def get_taxi_distance(self, pos_x, pos_y):
        return abs(pos_x[0] - pos_y[0]) + abs(pos_x[1] - pos_y[1])

    @torch.no_grad()
    def get_pairing_reward_from_scheduling(self, pairing, position, new_position, dest, dest_id):
        pairing_rewards = np.zeros((self.num_agents))
        for r in range(self.all_args.num_P):
            if pairing[r] != -1: # paired routing agent
                paired_l = pairing[r]
                old_dist = self.get_taxi_distance(position[r], position[paired_l])
                new_dist = self.get_taxi_distance(new_position[r], new_position[paired_l])
                if (new_position[r, 0] == new_position[paired_l, 0]) and (new_position[r, 1] == new_position[paired_l, 1]): # refuelling
                    # successfully accomplished the schedule for refuelling -- reward
                    pairing_rewards[r] += 0.05
                    pairing_rewards[paired_l] += 0.05

                elif new_dist < old_dist:
                    # the paired agents are getting closer, so give partial reward
                    if old_dist - new_dist == 2:
                        pairing_rewards[r] += 0.025
                        pairing_rewards[paired_l] += 0.025
                    elif old_dist - new_dist == 1:
                        pairing_rewards[r] += 0.0125
                        pairing_rewards[paired_l] += 0.0125
                else:
                    # paired but not refueling -- previously -0.05, now -0.025
                    pairing_rewards[r] -= 0.025
                    pairing_rewards[paired_l] -= 0.025


        return pairing_rewards

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        if self.algorithm_name == 'hetgat_mappo':
            value, action, action_log_prob, rnn_state, rnn_obs, rnn_state_critic, _ \
                = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                                np.concatenate(self.buffer.obs[step]),
                                                np.concatenate(self.buffer.rnn_states[step]),
                                                np.concatenate(self.buffer.rnn_obs[step]),
                                                np.concatenate(self.buffer.rnn_states_critic[step]),
                                                np.concatenate(self.buffer.masks[step]),
                                                np.concatenate(self.buffer.available_actions[step]))

            values = np.array(np.split(_t2n(value), self.n_rollout_threads))
            actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
            action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
            rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
            if self.all_args.tensor_obs or not self.all_args.binarized_obs_comm:
                rnn_obs = np.array(np.split(_t2n(rnn_obs), self.n_rollout_threads))
            rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

            return values, actions, action_log_probs, rnn_states, rnn_obs, rnn_states_critic
        else:
            value, action, action_log_prob, rnn_states, rnn_states_critic \
                = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                                  np.concatenate(self.buffer.obs[step]),
                                                  np.concatenate(self.buffer.rnn_states[step]),
                                                  np.concatenate(self.buffer.rnn_states_critic[step]),
                                                  np.concatenate(self.buffer.masks[step]))
            # [self.envs, agents, dim]
            values = np.array(np.split(_t2n(value), self.n_rollout_threads))
            actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
            action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
            rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
            rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
            # rearrange action
            if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.envs.action_space[0].shape):
                    uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                    if i == 0:
                        actions_env = uc_actions_env
                    else:
                        actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
            elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
                actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
            else:
                raise NotImplementedError
            return values, actions, action_log_probs, rnn_states, None, rnn_states_critic


    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, values, actions, action_log_probs, rnn_states, rnn_obs, rnn_states_critic = data


        if self.algorithm_name == 'hetgat_mappo':
            dones_env = np.all(dones, axis=1)
            if self.all_args.use_LSTM:

                if self.all_args.tensor_obs:

                    rnn_states = np.expand_dims(rnn_states, axis=2)
                    rnn_obs = np.expand_dims(rnn_obs, axis=2)


                    rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents * 2,
                                                            self.recurrent_N, 128), dtype=np.float32)

                    rnn_obs[dones_env == True] = np.zeros(((dones_env == True).sum(), (self.all_args.num_P + self.all_args.num_A) * 2,
                                                            self.recurrent_N, 16), dtype=np.float32)


                else:
                    if not self.all_args.binarized_obs_comm:
                        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents * 2,
                                                                self.recurrent_N, self.hidden_size),  dtype=np.float32)
                        rnn_obs[dones_env == True] = np.zeros(((dones_env == True).sum(), (self.all_args.num_P + self.all_args.num_A) * 2,
                                                            self.recurrent_N, 16), dtype=np.float32)


                rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(),
                                                                self.num_agents, *self.buffer.rnn_states_critic.shape[3:]),
                                                                dtype=np.float32)

            else:
                rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size),
                                                        dtype=np.float32)
                rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]),
                                                                dtype=np.float32)

            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
            active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in infos])

            if not self.use_centralized_V:
                share_obs = obs

            self.buffer.insert(share_obs, obs, rnn_states, rnn_obs, rnn_states_critic,
                               actions, action_log_probs, values, rewards, masks, bad_masks, active_masks, available_actions)
        else:
            rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
                                                 dtype=np.float32)
            rnn_states_critic[dones == True] = np.zeros(
                ((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

            self.buffer.insert_rmappo(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values,
                               rewards, masks)



    def log_train(self, train_infos, total_num_steps, total_time):

        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)
            self.logger.log_stat(k, v, total_num_steps, total_time)

    def log_env(self, env_infos, total_num_steps, total_time):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
                self.logger.log_stat(k, np.mean(v), total_num_steps, total_time)
        # self.logger.save_stats()    

    @torch.no_grad()
    def eval(self, total_num_steps=0, total_time=0):
        eval_battles_won = 0
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = []
        eval_ep_lengths = []
        eval_winning_ep_lengths = []

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()
        if self.all_args.use_LSTM:
            if self.all_args.tensor_obs:
                eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents * 2, self.recurrent_N, 128), dtype=np.float32)
                eval_rnn_obs = np.zeros((self.n_eval_rollout_threads, self.num_agents * 2, self.recurrent_N, 16), dtype=np.float32)
            else:
                if not self.all_args.binarized_obs_comm:
                    eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents * 2, self.recurrent_N, self.hidden_size), dtype=np.float32)
                    eval_rnn_obs = None
                else: # rnn_obs does not affect in binarized obs comm scenarios
                    eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents * 2, self.recurrent_N, self.hidden_size), dtype=np.float32)
                    eval_rnn_obs = np.zeros((self.n_eval_rollout_threads, self.num_agents * 2, self.recurrent_N, 16), dtype=np.float32)

            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents * 2, 1), dtype=np.float32)
        else:
            eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            if self.all_args.binarized_obs_comm:
                eval_rnn_obs = np.zeros((self.n_eval_rollout_threads, self.num_agents * 2, self.recurrent_N, 16), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        frame_num = 0
        eval_infos = self.eval_envs.get_info()
        eval_infos = [eval_infos]

        while True:

            if hasattr(self.all_args, 'plot_video') and self.all_args.plot_video:
                # generate frame
                self.generate_and_save_frame(eval_infos[0][0], frame_num)

            self.trainer.prep_rollout()

            if self.algorithm_name == 'hetgat_mappo':
                eval_actions, eval_rnn_states, eval_rnn_obs = \
                    self.trainer.policy.act(
                        # np.concatenate(eval_share_obs),
                        np.concatenate(eval_obs),
                        np.concatenate(eval_rnn_states),
                        np.concatenate(eval_rnn_obs),
                        np.concatenate(eval_masks),
                        np.concatenate(eval_available_actions),
                        deterministic=True)
            else:
                eval_actions, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                                       np.concatenate(eval_rnn_states),
                                                                       np.concatenate(eval_masks),
                                                                       deterministic=True)

            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            if self.algorithm_name == 'hetgat_mappo':
                if self.all_args.tensor_obs or not self.all_args.binarized_obs_comm:
                    eval_rnn_obs = np.array(np.split(_t2n(eval_rnn_obs), self.n_eval_rollout_threads))
            # rnn_obs does not affect in binarized obs comm scenarios

            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)
            one_episode_rewards.append(eval_rewards)

            frame_num += 1


            eval_dones_env = np.all(eval_dones, axis=1)

            if self.all_args.use_LSTM:
                if self.all_args.tensor_obs:
                    eval_rnn_states[eval_dones_env == True] = np.zeros(
                        ((eval_dones_env == True).sum(), self.num_agents * 2, self.recurrent_N, 128), dtype=np.float32).reshape(*eval_rnn_states[eval_dones_env == True].shape)
                    eval_rnn_obs[eval_dones_env == True] = np.zeros(
                        ((eval_dones_env == True).sum(), self.num_agents * 2, self.recurrent_N, 16), dtype=np.float32).reshape(*eval_rnn_obs[eval_dones_env == True].shape)
                else:
                    eval_rnn_states[eval_dones_env == True] = np.zeros(
                        ((eval_dones_env == True).sum(), self.num_agents * 2, self.recurrent_N, self.hidden_size), dtype=np.float32)
                    if not self.all_args.binarized_obs_comm:
                        eval_rnn_obs[eval_dones_env == True] = np.zeros(
                            ((eval_dones_env == True).sum(), self.num_agents * 2, self.recurrent_N, 16), dtype=np.float32)
                    else: # rnn_obs does not affect in binarized obs comm scenarios
                        eval_rnn_obs = np.zeros((self.n_eval_rollout_threads, self.num_agents * 2, self.recurrent_N, 16), dtype=np.float32)


                eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents * 2, 1), dtype=np.float32)
                eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents * 2, 1), dtype=np.float32)

            else:
                eval_rnn_states[eval_dones_env == True] = np.zeros(
                    ((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

                eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
                eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)


            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    if hasattr(self.all_args, 'plot_video') and self.all_args.plot_video:
                        print('%d eval episodes done, stored frames: %d'%(eval_episode, frame_num))
                    eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                    eval_ep_lengths.append(eval_infos[eval_i][0]["episode_steps"])
                    one_episode_rewards = []
                    if eval_infos[eval_i][0]['won']:
                        eval_battles_won += 1
                        eval_winning_ep_lengths.append(eval_infos[eval_i][0]["episode_steps"])


            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.array(eval_episode_rewards)
                eval_episode_lengths = np.array(eval_ep_lengths)
                eval_win_rate = np.array([eval_battles_won / eval_episode])
                eval_winning_episode_lengths = np.array(eval_winning_ep_lengths)
                eval_env_infos = {'eval_average_episode_rewards': eval_episode_rewards,
                                  'eval_average_episode_lengths': eval_episode_lengths,
                                  'eval_winning_average_episode_lengths': eval_winning_episode_lengths,
                                  'eval_win_rate': eval_win_rate}

                self.log_env(eval_env_infos, total_num_steps, total_time)

                print("eval rewards is {}.".format(np.mean(eval_episode_rewards)))
                print("eval win rate is {}.".format(eval_win_rate))
                print(f"eval ep length is {np.mean(eval_episode_lengths)}")
                if eval_battles_won > 0:
                    print(f"eval winning ep length is {np.mean(eval_winning_episode_lengths)}")
                break

        # if plot_video, generate a video
        if hasattr(self.all_args, 'plot_video') and self.all_args.plot_video:
            # generate the last frame
            self.generate_and_save_frame(eval_infos[0][0], frame_num)
            # generate a video
            scenario = self.all_args.model_dir.split('results/')[1]
            scenario = 'videos/' + scenario
            frame_array = []
            for f_num in range(frame_num+1):
                img = cv2.imread('%s/frame_%d.png'%(scenario, f_num))
                height, width, layers = img.shape
                size = (width, height)
                frame_array.append(img)

            out = cv2.VideoWriter('%s/video.avi' % scenario, cv2.VideoWriter_fourcc(*'DIVX'), 2, size)

            for i in range(len(frame_array)):
                out.write(frame_array[i])
            out.release()


    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        if self.algorithm_name == 'hetgat_mappo':
            next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                         np.concatenate(self.buffer.obs[-1]),
                                                         np.concatenate(self.buffer.rnn_states[-1]),
                                                         np.concatenate(self.buffer.rnn_obs[-1]),
                                                         np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                         np.concatenate(self.buffer.masks[-1]),
                                                         np.concatenate(self.buffer.available_actions[-1]))
        else:
            next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                         np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                         np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    @torch.no_grad()
    def generate_and_save_frame(self, dict_state, f_num):
        font = cv2.FONT_HERSHEY_COMPLEX
        fontscale = 1
        fontcolor = (0, 0, 0)
        thickness = 2

        dest_pos_list = dict_state['dest']
        routing_locs = dict_state['locs'][:self.all_args.num_P]
        routing_locs_array = np.array(routing_locs)
        logis_locs = dict_state['locs'][self.all_args.num_P:]
        logis_locs_array = np.array(logis_locs)
        intensity = dict_state['intensity']
        fuel_left = dict_state['fuel_left']
        curr_step = dict_state['current_step']

        canvas = np.ones((100*self.all_args.dim + 200, 100*self.all_args.dim + 200, 3)) * 255


        # Color the frame with intensity
        # Color interpolation between (60, 80, 200)(intensity=0) to (230, 245, 255)(intensity=1)
        for i in range(self.all_args.dim):
            for j in range(self.all_args.dim):
                intensity_pos = intensity[i][j]
                canvas[100*i+100:100*i+200, 100*j+100:100*j+200, 0] = intensity_pos*230 + (1-intensity_pos)*60
                canvas[100*i+100:100*i+200, 100*j+100:100*j+200, 1] = intensity_pos*245 + (1-intensity_pos)*80
                canvas[100*i+100:100*i+200, 100*j+100:100*j+200, 2] = intensity_pos*255 + (1-intensity_pos)*200

        # Put routing agents and logistics agents
        # read the icon
        icon_size = 100
        routing_img = cv2.imread("routing.png", cv2.IMREAD_UNCHANGED)
        routing_icon_mask_orig = routing_img[:, :, 3]
        routing_icon_mask = cv2.resize(routing_icon_mask_orig, (icon_size, icon_size))
        routing_half_icon_mask = cv2.resize(routing_icon_mask_orig, (int(icon_size/2), icon_size))
        logis_img = cv2.imread("logistics.png", cv2.IMREAD_UNCHANGED)
        logis_icon_mask_orig = logis_img[:, :, 3]
        logis_icon_mask = cv2.resize(logis_icon_mask_orig, (icon_size, icon_size))
        logis_half_icon_mask = cv2.resize(logis_icon_mask_orig, (int(icon_size/2), icon_size))

        all_locs = dict_state['locs']
        uniq_locs = np.unique(all_locs, axis=0)

        for loc in uniq_locs:

            original_patch = canvas[100*loc[0]+100:100*loc[0]+200, 100*loc[1]+100:100*loc[1]+200]
            routing_num = np.count_nonzero(np.all(routing_locs_array == loc, axis=1))
            logis_num = np.count_nonzero(np.all(logis_locs_array == loc, axis=1))
            if routing_num == 0: #only logistic agent is there so put logistic agent as orange and put the number of logistic agents
                canvas[100*loc[0]+100:100*loc[0]+200, 100*loc[1]+100:100*loc[1]+200, 0] = np.where(logis_icon_mask > 0, 255, original_patch[:, :, 0])
                canvas[100*loc[0]+100:100*loc[0]+200, 100*loc[1]+100:100*loc[1]+200, 1] = np.where(logis_icon_mask > 0, 215, original_patch[:, :, 1])
                canvas[100*loc[0]+100:100*loc[0]+200, 100*loc[1]+100:100*loc[1]+200, 2] = np.where(logis_icon_mask > 0, 0, original_patch[:, :, 2])
                canvas = cv2.putText(canvas, '%d'%logis_num, [100*loc[1]+150, 100*loc[0]+150], font, fontscale, fontcolor, thickness, cv2.LINE_AA, False)

            elif logis_num == 0: #only routing agent is there so put routing agent as dark purple and put the number of routing agents
                canvas[100*loc[0]+100:100*loc[0]+200, 100*loc[1]+100:100*loc[1]+200, 0] = np.where(routing_icon_mask > 0, 135, original_patch[:, :, 0])
                canvas[100*loc[0]+100:100*loc[0]+200, 100*loc[1]+100:100*loc[1]+200, 1] = np.where(routing_icon_mask > 0, 31, original_patch[:, :, 1])
                canvas[100*loc[0]+100:100*loc[0]+200, 100*loc[1]+100:100*loc[1]+200, 2] = np.where(routing_icon_mask > 0, 120, original_patch[:, :, 2])
                canvas = cv2.putText(canvas, '%d'%routing_num, [100*loc[1]+150, 100*loc[0]+150], font, fontscale, fontcolor, thickness, cv2.LINE_AA, False)

            else: #routing agent and logistics agent are together so put both
                # print(logis_half_icon_mask.shape, canvas[100*loc[0]+100:100*loc[0]+200, 100*loc[1]+100:100*loc[1]+150, 0].shape, original_patch[:, :50, 0].shape)
                canvas[100*loc[0]+100:100*loc[0]+200, 100*loc[1]+100:100*loc[1]+150, 0] = np.where(logis_half_icon_mask > 0, 255, original_patch[:, :50, 0])
                canvas[100*loc[0]+100:100*loc[0]+200, 100*loc[1]+100:100*loc[1]+150, 1] = np.where(logis_half_icon_mask > 0, 215, original_patch[:, :50, 1])
                canvas[100*loc[0]+100:100*loc[0]+200, 100*loc[1]+100:100*loc[1]+150, 2] = np.where(logis_half_icon_mask > 0, 0, original_patch[:, :50, 2])
                canvas = cv2.putText(canvas, '%d'%logis_num, [100*loc[1]+125, 100*loc[0]+150], font, fontscale, fontcolor, thickness, cv2.LINE_AA, False)
                canvas[100*loc[0]+100:100*loc[0]+200, 100*loc[1]+150:100*loc[1]+200, 0] = np.where(routing_half_icon_mask > 0, 135, original_patch[:, 50:, 0])
                canvas[100*loc[0]+100:100*loc[0]+200, 100*loc[1]+150:100*loc[1]+200, 1] = np.where(routing_half_icon_mask > 0, 31, original_patch[:, 50:, 1])
                canvas[100*loc[0]+100:100*loc[0]+200, 100*loc[1]+150:100*loc[1]+200, 2] = np.where(routing_half_icon_mask > 0, 120, original_patch[:, 50:, 2])
                canvas = cv2.putText(canvas, '%d'%routing_num, [100*loc[1]+175, 100*loc[0]+150], font, fontscale, fontcolor, thickness, cv2.LINE_AA, False)


        # Draw grid
        # draw vertical lines
        for x in range(self.all_args.dim + 1):
            canvas = cv2.line(canvas, (100*x+100, 100), (100*x+100, 100*self.all_args.dim+100), color=(180, 180, 180), thickness=1)
            canvas = cv2.line(canvas, (100, 100*x+100), (100*self.all_args.dim+100, 100*x+100), color=(180, 180, 180), thickness=1)



        # Color the destination position as red
        for dest_pos in dest_pos_list:
            canvas_dest_pos = [100 * dest_pos[0], 100 * dest_pos[1]]  # (x, y)
            canvas[canvas_dest_pos[0]+100: canvas_dest_pos[0]+200, canvas_dest_pos[1]+100:canvas_dest_pos[1]+200, 0] = 255
            canvas[canvas_dest_pos[0]+100: canvas_dest_pos[0]+200, canvas_dest_pos[1]+100:canvas_dest_pos[1]+200, 1] = 0
            canvas[canvas_dest_pos[0]+100: canvas_dest_pos[0]+200, canvas_dest_pos[1]+100:canvas_dest_pos[1]+200, 2] = 0

        # Write down the current step number and whether success or not
        info_str = 'Current step: %d, Fuel: '%curr_step
        for i in fuel_left:
            info_str = info_str + '%d, '%i
        info_str = info_str[:-2]
        canvas = cv2.putText(canvas, info_str, [50, 50], font, fontscale, (1,1,1), thickness, cv2.LINE_AA, False)

        scenario = self.all_args.model_dir.split('results/')[1]
        scenario = 'videos/' + scenario
        if not Path(scenario).exists():
            os.makedirs(scenario)

        # Save image
        cv2.imwrite('%s/frame_%d.png'%(scenario, f_num), canvas[:,:,::-1])
        return canvas
