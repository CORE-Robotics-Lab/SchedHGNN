from ast import arg
from tkinter.messagebox import NO
from turtle import forward
import torch
import torch.nn as nn

import numpy as np

import copy

from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.cnn import CNNBase
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.algorithms.utils.hetgat import HetGATLayer
from onpolicy.algorithms.utils.rnn import LSTMLayer
from onpolicy.algorithms.utils.convlstm import Seq2Seq
from onpolicy.utils.util import get_shape_from_obs_space
from onpolicy.algorithms.hetgat_mappo.graphs.fastreal import MultiHeteroGATLayerReal

class GAT_Actor(nn.Module):
    """
    Actor network class for MAPPO with graph attention. 
    Outputs actions given observations.
    
    Args:
        # TODO: write better argument description
        nn (_type_): _description_
    """

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(GAT_Actor, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.hetnet_hidden_size = args.hetnet_hidden_dim
        self.hetnet_num_heads = args.hetnet_num_heads
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.args.upsample_x2 = False


        if self.args.tensor_obs:

            if self.args.vision == 1 or self.args.vision == 2:

                if 'Marine' in self.args.env_name:
                    self.num_obs_in_channels = 4
                else:
                    raise NotImplementedError


                self.prepro_obs = nn.Sequential(
                    nn.Conv2d(in_channels=self.num_obs_in_channels, out_channels=16, kernel_size=(2, 2)),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )


            else:
                self.prepro_obs = nn.Sequential(
                    nn.Flatten(1, -1),
                    nn.Linear(3, 6),
                    nn.Linear(6, 16)
                )

            self.num_stat_in_channels = 1

            self.prepro_stat = nn.Sequential(
                nn.Conv2d(in_channels=self.num_stat_in_channels, out_channels=64, kernel_size=(3, 3)),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2)),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )

            self.f_module_obs = nn.LSTMCell(16, 16)
            self.f_module_stat = nn.LSTMCell(128, 128)
            input_size = self.args.hidden_size
            if self.args.upsample_x2:
                self.upsample_stat = nn.Upsample(scale_factor=2, mode='nearest')


        else:
            obs_shape = get_shape_from_obs_space(obs_space)
            if args.binarized_obs_comm:
                input_size = obs_shape[0]
                if args.env_name == 'Marine' and args.schedule_rewards:
                    input_size += self.args.num_P + self.args.num_A

            else:
                base = CNNBase if len(obs_shape) == 3 else MLPBase
                input_size = self.args.hidden_size
                self.base = base(args, obs_shape)
            if args.use_LSTM:
                self.rnn = LSTMLayer(args, self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            else:
                self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)


        self.gat = HetGATLayer(args, self.hidden_size, device=device, input_size=input_size)
        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)
        self.critic = nn.Linear(9, 1)
        self.relu = nn.ReLU()
        self.to(device)

        self.action_have_vision = ('Marine' in self.args.env_name)
        print('gat_actor_critic, action_have_vision', self.action_have_vision)
        
    def forward(self, obs, cent_obs, rnn_states, rnn_obs, masks,
                available_actions=None, deterministic=False, graph=None, batch_size=1):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param cent_obs: global state input for SSN, shoudl be disconnected during execution
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        if cent_obs is not None:
            cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)


        if self.args.tensor_obs:
            if self.args.binarized_obs_comm:
                x_per_stat, x_act_stat, x_per_obs, x_act_obs = self.marine_get_states_obs_from_binarized_obs(obs)

            else:
                x_per_stat, x_act_stat, x_per_obs, x_act_obs = self.get_states_obs_from_tensor(obs)


            x_per_stat, x_act_stat, x_per_obs, x_act_obs = check(x_per_stat).to(**self.tpdv), check(x_act_stat).to(**self.tpdv), \
                                                check(x_per_obs).to(**self.tpdv), check(x_act_obs).to(**self.tpdv)

            if self.args.upsample_x2:
                x_per_stat = self.upsample_stat(x_per_stat[None, :])  # Send to upsample layer with extra batch dimension
                x_per_stat = torch.squeeze(x_per_stat, 0)  # Remove extra batch dimension

                if self.args.num_A != 0:
                    x_act_stat = self.upsample_stat(x_act_stat[None, :])
                    x_act_stat = torch.squeeze(x_act_stat, 0)


            state_per_stat = self.prepro_stat(x_per_stat)

            if self.args.num_A != 0:
                state_act_stat = self.prepro_stat(x_act_stat)

            if self.args.vision == 0 or self.args.vision == 1 or self.args.vision == 2:
                x_per_obs = self.prepro_obs(x_per_obs)
            else:
                x_per_obs = self.prepro_obs(x_per_obs.squeeze())

            if self.action_have_vision:
                ### we assume action agents has vision, A_vision==1 so action agents have obs
                x_act_obs = self.prepro_obs(x_act_obs)



        else:
            obs = check(obs).to(**self.tpdv)
            if self.args.binarized_obs_comm and self.args.env_name == 'Marine' and self.args.schedule_rewards:
                comm_vectors = self.marine_get_schedules_from_comm_vectors(obs)
            elif self.args.binarized_obs_comm:
                comm_vectors = obs
            else:
                comm_vectors = self.base(obs)



        # to LSTM
        if self.args.use_LSTM:

            if self.args.tensor_obs:
                rnn_states = rnn_states.squeeze()
                rnn_obs = check(rnn_obs).to(**self.tpdv)
                # Forward pass LSTM features associated with the state of the perception agents
                hidden_state, cell_state = torch.split(rnn_states, [self.args.num_P + self.args.num_A] * 2)
                hidden_state_per_stat, cell_state_per_stat = hidden_state[:self.args.num_P], \
                                                             cell_state[:self.args.num_P]
                hidden_state_per_stat, cell_state_per_stat = self.f_module_stat(state_per_stat.squeeze(),
                                                                                (hidden_state_per_stat,
                                                                                 cell_state_per_stat))

                if self.args.num_A != 0:

                    # Forward pass LSTM features associated with the state of the action agents
                    hidden_state_act_stat, cell_state_act_stat = hidden_state[self.args.num_P:],\
                                                                 cell_state[self.args.num_P:]
                    hidden_state_act_stat, cell_state_act_stat = self.f_module_stat(
                        state_act_stat.squeeze().reshape(self.args.num_A, -1), # in case only one action agent
                        (hidden_state_act_stat,
                         cell_state_act_stat))

                # forward pass the LSTM features associated with the observations of the perception agents
                # hidden_state_per_obs, cell_state_per_obs = torch.split(rnn_obs, [self.args.num_P] * 2)
                ### action agents also have vision
                if self.action_have_vision and self.args.num_A != 0:
                    hidden_state_obs, cell_state_obs = torch.split(rnn_obs, [self.args.num_P + self.args.num_A] * 2)
                    hidden_state_per_obs, cell_state_per_obs = hidden_state_obs[:self.args.num_P], \
                                                                 cell_state_obs[:self.args.num_P]
                    hidden_state_per_obs, cell_state_per_obs = self.f_module_obs(x_per_obs.squeeze(),
                                                                                 (hidden_state_per_obs.reshape(-1, 16),
                                                                                  cell_state_per_obs.reshape(-1, 16)))

                    ### action agent also has observation
                    # Forward pass LSTM features associated with the state of the action agents
                    hidden_state_act_obs, cell_state_act_obs = hidden_state_obs[self.args.num_P:], \
                                                                 cell_state_obs[self.args.num_P:]
                    hidden_state_act_obs, cell_state_act_obs = self.f_module_obs(
                        x_act_obs.squeeze().reshape(self.args.num_A, -1),  # in case only one action agent
                        (hidden_state_act_obs.reshape(-1, 16),
                         cell_state_act_obs.reshape(-1, 16)))



                else:
                    hidden_state_per_obs, cell_state_per_obs = torch.split(rnn_obs, [self.args.num_P] * 2)
                    hidden_state_per_obs, cell_state_per_obs = self.f_module_obs(x_per_obs.squeeze(),
                                                                                 (hidden_state_per_obs.reshape(-1, 16),
                                                                                  cell_state_per_obs.reshape(-1, 16)))

                if self.args.num_A != 0:
                    rnn_states = torch.cat((hidden_state_per_stat, hidden_state_act_stat,
                                            cell_state_per_stat, cell_state_act_stat))
                else:
                    rnn_states = torch.cat((hidden_state_per_stat, cell_state_per_stat))


                # action agents also have vision
                if self.action_have_vision and (self.args.num_A != 0):
                    rnn_obs = torch.cat((hidden_state_per_obs, hidden_state_act_obs,
                                         cell_state_per_obs, cell_state_act_obs))
                else:
                    rnn_obs = torch.cat((hidden_state_per_obs, cell_state_per_obs))

                feat_dict = {}

                feat_dict['P'] = torch.cat([hidden_state_per_stat, hidden_state_per_obs], dim=1)
                if self.action_have_vision and self.args.num_A != 0:
                    # action agents also have vision
                    feat_dict['A'] = torch.cat([hidden_state_act_stat, hidden_state_act_obs], dim=1)
                elif self.args.num_A != 0:
                    feat_dict['A'] = hidden_state_act_stat



            else:
                if self.args.binarized_obs_comm:
                    feat_dict = {}
                    feat_dict['P'] = comm_vectors[:self.args.num_P]
                    feat_dict['A'] = comm_vectors[self.args.num_P:]


        else:
            if self.args.binarized_obs_comm:
                feat_dict = {}
                feat_dict['P'] = comm_vectors[:self.args.num_P]
                feat_dict['A'] = comm_vectors[self.args.num_P:]


        actor_features, state_features = self.gat(feat_dict, cent_obs, graph, batch_size)
        if self.args.tensor_obs:
            actor_features = actor_features.squeeze()

        elif self.args.binarized_obs_comm:
            actor_features, rnn_states = self.rnn(actor_features.squeeze(), rnn_states) # rnn after communication


        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)
        all_action_log_probs = self.act.get_logits(actor_features, available_actions)

        return actions, action_log_probs, rnn_states, rnn_obs, state_features, all_action_log_probs




    def get_states_obs_from_tensor(self, x):
        # print(len(x.shape)) # 2
        if len(x.shape) == 2: # if this is for a sample
            # action agents have vision as well
            x_per_stat = []
            x_act_stat = []
            x_per_obs = []
            x_act_obs = []

            for i in range(self.args.num_P):
                x_per_stat.append(x[i][1])
                x_per_obs.append(x[i][0])
            for i in range(self.args.num_P, self.args.num_P + self.args.num_A):
                x_act_stat.append(x[i][1])
                ## this part is added
                if self.action_have_vision:
                    x_act_obs.append(x[i][0])


            return torch.tensor(np.array(x_per_stat)), torch.tensor(np.array(x_act_stat)),\
                   torch.tensor(np.array(x_per_obs)), torch.tensor(np.array(x_act_obs))

        else: # if this is for a batch of samples
            x = list(x)
            get_x_per_stat = lambda x_in: [x_in[i][1]for i in range(self.args.num_P)]
            get_x_act_stat = lambda x_in: [x_in[i][1]for i in range(self.args.num_P, self.args.num_P + self.args.num_A)]
            get_x_per_obs = lambda x_in: [x_in[i][0] for i in range(self.args.num_P)]
            x_per_stat = np.stack(list(map(get_x_per_stat, x)))
            x_act_stat = np.stack(list(map(get_x_act_stat, x)))
            x_per_obs = np.stack(list(map(get_x_per_obs, x)))

            if self.action_have_vision:
            # in our setting, self.A_vision = 1 so we have observation for action agents
                get_x_act_obs = lambda x_in: [x_in[i][0] for i in range(self.args.num_P, self.args.num_P + self.args.num_A)]
                x_act_obs = np.stack(list(map(get_x_act_obs, x)))

                return torch.unsqueeze(torch.tensor(np.array(x_per_stat)), dim=2), \
                       torch.unsqueeze(torch.tensor(np.array(x_act_stat)), dim=2), \
                       torch.unsqueeze(torch.tensor(np.array(x_per_obs)), dim=2), \
                       torch.unsqueeze(torch.tensor(np.array(x_act_obs)), dim=2)
            else:
                x_act_obs = []
                return torch.unsqueeze(torch.tensor(np.array(x_per_stat)), dim=2), \
                       torch.unsqueeze(torch.tensor(np.array(x_act_stat)), dim=2), \
                       torch.unsqueeze(torch.tensor(np.array(x_per_obs)), dim=2), \
                       torch.Tensor(x_act_obs)



    def evaluate_actions(self, obs, cent_obs, rnn_states, rnn_obs, action, masks, available_actions=None,
                         active_masks=None,graph_batch=None,batch_size=1):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param cent_obs: global state input for SSN, shoudl be disconnected during execution 
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """


        if cent_obs is not None:
            cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)





        if self.args.tensor_obs:
            if self.args.binarized_obs_comm:
                x_per_stat, x_act_stat, x_per_obs, x_act_obs = self.marine_get_states_obs_from_binarized_obs(obs)
            else:
                x_per_stat, x_act_stat, x_per_obs, x_act_obs = self.get_states_obs_from_tensor(obs)
            x_per_stat, x_act_stat, x_per_obs, x_act_obs = check(x_per_stat).to(**self.tpdv), check(x_act_stat).to(**self.tpdv), \
                                                check(x_per_obs).to(**self.tpdv), check(x_act_obs).to(**self.tpdv)


            state_per_stat = self.prepro_stat(x_per_stat.reshape(-1, *x_per_stat.shape[2:]))
            if self.args.num_A != 0:
                state_act_stat = self.prepro_stat(x_act_stat.reshape(-1, *x_act_stat.shape[2:]))


            x_per_obs = self.prepro_obs(x_per_obs.reshape(-1, *x_per_obs.shape[2:]))
            if self.action_have_vision:
                x_act_obs = self.prepro_obs(x_act_obs.reshape(-1, *x_act_obs.shape[2:]))

        else:
            obs = check(obs).to(**self.tpdv)
            if self.args.binarized_obs_comm and self.args.env_name == 'Marine' and self.args.schedule_rewards:
                comm_vectors = self.marine_get_schedules_from_comm_vectors(obs)
            elif self.args.binarized_obs_comm:
                comm_vectors = obs
            else:
                comm_vectors = self.base(obs)
        # to LSTM
        if self.args.use_LSTM:

            if self.args.tensor_obs:
                rnn_states = rnn_states.squeeze()
                rnn_obs = check(rnn_obs).to(**self.tpdv)
                rnn_obs = rnn_obs.squeeze()

                # Forward pass LSTM features associated with the state of the perception agents
                hidden_state, cell_state = torch.split(rnn_states, [self.args.num_P + self.args.num_A] * 2, dim=1)


                hidden_state_per_stat, cell_state_per_stat = hidden_state[:, :self.args.num_P, :].reshape(-1, 128), \
                                                             cell_state[:, :self.args.num_P, :].reshape(-1, 128)


                hidden_state_per_stat, cell_state_per_stat = self.f_module_stat(state_per_stat.squeeze(),
                                                                                (hidden_state_per_stat,
                                                                                 cell_state_per_stat))


                if self.args.num_A != 0:
                    # Forward pass LSTM features associated with the state of the action agents
                    hidden_state_act_stat, cell_state_act_stat = hidden_state[:, self.args.num_P:, :].reshape(-1, 128), \
                                                                 cell_state[:, self.args.num_P:, :].reshape(-1, 128)

                    hidden_state_act_stat, cell_state_act_stat = self.f_module_stat(state_act_stat.squeeze(),
                                                                                    (hidden_state_act_stat,
                                                                                     cell_state_act_stat))

                # forward pass the LSTM features associated with the observations of the perception agents
                # hidden_state_per_obs, cell_state_per_obs = torch.split(rnn_obs, [self.args.num_P] * 2, dim=1)
                if self.action_have_vision and self.args.num_A != 0:
                    hidden_state_obs, cell_state_obs = torch.split(rnn_obs, [self.args.num_P + self.args.num_A] * 2, dim=1)
                    hidden_state_per_obs, cell_state_per_obs = hidden_state_obs[:, :self.args.num_P, :].reshape(-1, 16), \
                                                                 cell_state_obs[:, :self.args.num_P, :].reshape(-1, 16)

                    hidden_state_per_obs, cell_state_per_obs = self.f_module_obs(x_per_obs.squeeze(),
                                                                                 (hidden_state_per_obs,
                                                                                  cell_state_per_obs))

                    # action agent also has observation
                    # forward pass the LSTM features associated with the observations of the action agents
                    hidden_state_act_obs, cell_state_act_obs = hidden_state_obs[:, self.args.num_P:, :].reshape(-1, 16), \
                                                               cell_state_obs[:, self.args.num_P:, :].reshape(-1, 16)

                    hidden_state_act_obs, cell_state_act_obs = self.f_module_obs(x_act_obs.squeeze(),
                                                                                 (hidden_state_act_obs,
                                                                                  cell_state_act_obs))
                else:
                    hidden_state_per_obs, cell_state_per_obs = torch.split(rnn_obs, [self.args.num_P] * 2, dim=1)
                    hidden_state_per_obs, cell_state_per_obs = hidden_state_per_obs.reshape(-1, 16), cell_state_per_obs.reshape(-1, 16)
                    hidden_state_per_obs, cell_state_per_obs = self.f_module_obs(x_per_obs.squeeze(),
                                                                                 (hidden_state_per_obs,
                                                                                  cell_state_per_obs))

                feat_dict = {}

                feat_dict['P'] = torch.cat([hidden_state_per_stat, hidden_state_per_obs], dim=1) # concat to 128 + 16 -> 144


                if self.action_have_vision and self.args.num_A != 0:
                    # action agents also have vision
                    feat_dict['A'] = torch.cat([hidden_state_act_stat, hidden_state_act_obs], dim=1)
                elif self.args.num_A != 0:
                    feat_dict['A'] = hidden_state_act_stat



            else:
                if self.args.binarized_obs_comm:
                    # if binarized_obs_comm, even though use_LSTM=True, we don't use any LSTM
                    # actor_features is kept

                    feat_dict = {}
                    obs_hidden_size = comm_vectors.shape[-1]
                    feat_dict['P'] = comm_vectors[:, :self.args.num_P].reshape((-1, obs_hidden_size))
                    feat_dict['A'] = comm_vectors[:, self.args.num_P:].reshape((-1, obs_hidden_size))


        else:
            if self.args.binarized_obs_comm:
                # if binarized_obs_comm, even though use_LSTM=True, we don't use any LSTM
                # actor_features is kept
                feat_dict = {}
                obs_hidden_size = comm_vectors.shape[-1]
                feat_dict['P'] = comm_vectors[:, :self.args.num_P].reshape((-1, obs_hidden_size))
                feat_dict['A'] = comm_vectors[:, self.args.num_P:].reshape((-1, obs_hidden_size))


        actor_features, state_features = self.gat(feat_dict, cent_obs, graph_batch, batch_size)
        if self.args.tensor_obs:
            actor_features = actor_features.squeeze()
        elif self.args.binarized_obs_comm:
            actor_features, rnn_states = self.rnn(actor_features.squeeze(), rnn_states)  # rnn after communication

        actor_features = actor_features.reshape(actor_features.shape[0]*actor_features.shape[1], -1)
        


        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                   action, available_actions,
                                                                   active_masks=
                                                                   active_masks if self._use_policy_active_masks
                                                                   else None)
        all_action_log_probs = self.act.get_logits(actor_features, available_actions)

        return action_log_probs, dist_entropy, state_features, all_action_log_probs



    '''
    Used for the binarized observation & communication
    Should we convert binarized obs into tensor_obs in this code or accept tensorized obs from env & assume we already get the position and fuel level of all agents through communication?
    The former one - the bottom functions will be used, the latter one - the bottom functions will not be used
    '''
    def get_taxi_distance(self, pos_x, pos_y):
        return abs(pos_x[0] - pos_y[0]) + abs(pos_x[1] - pos_y[1])

    def binary_to_coordinate(self, binary, dim):
        # for getting coordinates from comm vectors
        decimal = self.binary_to_decimal(binary)

        x = int(decimal / dim)
        y = decimal % dim

        return [x, y]

    def binary_to_decimal(self, binary):

        n = binary[::-1]
        ones_locs = np.where(n==1)[0]
        decimal = 0
        for i in ones_locs:
            decimal += 2**i

        return decimal

    def marine_decode_agents_position_destination_fuel(self, comm_vectors):
        if len(comm_vectors.shape) < 3:
            batch_input = False
            comm_vectors = comm_vectors.reshape(1, *comm_vectors.shape)
        else:
            batch_input = True


        BASE = self.args.dim ** 2
        binary_BASE = BASE.bit_length()  # binarized value
        max_fuel = int(self.args.dim)
        binary_max_fuel = max_fuel.bit_length()

        if batch_input:
            batch_agent_positions = []
            batch_r_dests = []
            batch_fuel_levels = []


        for b in range(comm_vectors.shape[0]): # batch

            batch_comm_vectors = comm_vectors[b]
            agent_positions = []
            r_dests = []
            fuel_levels = []

            for i in range(self.args.num_P+self.args.num_A):

                agent_position = self.binary_to_coordinate(batch_comm_vectors[i, :binary_BASE], self.args.dim)
                agent_positions.append(agent_position)

                if i < self.args.num_P:
                    r_agent_destination = self.binary_to_coordinate(batch_comm_vectors[i, binary_BASE:2*binary_BASE], self.args.dim)
                    r_dests.append(r_agent_destination)
                    agent_fuel = self.binary_to_decimal(batch_comm_vectors[i, 2*binary_BASE:2*binary_BASE+binary_max_fuel])
                    fuel_levels.append(agent_fuel)
                elif self.args.limited_refuel: # Logistics agents have fuel_level in limited_refuel case
                    l_agent_destination = self.binary_to_coordinate(batch_comm_vectors[i, binary_BASE:2*binary_BASE], self.args.dim) # for the easier implementation (the dest info is given to all agents)
                    r_dests.append(l_agent_destination)
                    agent_fuel = self.binary_to_decimal(batch_comm_vectors[i, 2*binary_BASE:2*binary_BASE+binary_max_fuel])
                    fuel_levels.append(agent_fuel)

            if batch_input:
                batch_agent_positions.append(agent_positions)
                batch_r_dests.append(r_dests)
                batch_fuel_levels.append(fuel_levels)

        if batch_input:
            return batch_agent_positions, batch_r_dests, batch_fuel_levels
        else:
            return agent_positions, r_dests, fuel_levels

    def marine_get_schedules_from_comm_vectors(self, comm_vectors, batch_agent_positions, batch_r_dests, batch_fuel_levels):
        if len(comm_vectors.shape) < 3:
            batch_input = False
            comm_vectors = comm_vectors.reshape(1, *comm_vectors.shape)
        else:
            batch_input = True


        pairings = np.zeros((comm_vectors.shape[0], self.args.num_P + self.args.num_A))  # (b, num_agents)


        for b in range(comm_vectors.shape[0]): # batch

            routing_agents_that_need_pairing = []
            distance_to_dest_of_routing_agents_that_need_pairing = []
            logistics_agents_that_can_be_paired = np.array(list(range(self.args.num_P, self.args.num_P + self.args.num_A)))
            agent_positions = batch_agent_positions[b]
            r_positions = agent_positions[:self.args.num_P]
            l_positions = agent_positions[self.args.num_P:]
            r_dests = batch_r_dests[b]
            fuel_levels = batch_fuel_levels[b]

            for i in range(self.args.num_P):

                r_agent_position = agent_positions[i]
                r_agent_destination = r_dests[i]

                distance_to_dest = self.get_taxi_distance(r_agent_destination, r_agent_position)
                agent_fuel = fuel_levels[i]

                distance_to_dest_of_routing_agents_that_need_pairing.append(distance_to_dest)

                if distance_to_dest > agent_fuel:
                    routing_agents_that_need_pairing.append(i)

            routing_agents_that_need_pairing.sort(key=lambda x: fuel_levels[x], reverse=False)  # ascending, that means, the routing agents having low level of fuel will be paired first

            for r in routing_agents_that_need_pairing:
                r_agent_position = r_positions[r]
                if len(logistics_agents_that_can_be_paired) == 0:
                    break
                distance_to_remaining_logis = []
                for l in logistics_agents_that_can_be_paired:
                    l_agent_position = l_positions[l - self.args.num_P]
                    distance_to_logis = self.get_taxi_distance(r_agent_position, l_agent_position)
                    distance_to_remaining_logis.append(distance_to_logis)

                sorted_index = np.argsort(distance_to_remaining_logis) # ascending
                logistics_agents_that_can_be_paired = logistics_agents_that_can_be_paired[sorted_index] # ascending, that means, the logis agent that is closest to the routing agent will be paired

                paired_logis = logistics_agents_that_can_be_paired[0]
                pairings[b, r] = paired_logis+1
                pairings[b, paired_logis] = r+1
                logistics_agents_that_can_be_paired = logistics_agents_that_can_be_paired[1:]


        return pairings

    def marine_get_states_obs_from_binarized_obs(self, obs):

        comm_vectors = obs

        if len(comm_vectors.shape) < 3:
            batch_input = False
            agent_positions, r_dests, fuel_levels = self.marine_decode_agents_position_destination_fuel(comm_vectors)
            batch_agent_positions = [agent_positions]
            batch_r_dests = [r_dests]
            batch_fuel_levels = [fuel_levels]
            if self.args.schedule_rewards:
                pairings = self.marine_get_schedules_from_comm_vectors(comm_vectors, batch_agent_positions, batch_r_dests, batch_fuel_levels) # [b, num_agents]
            batch_size = 1
            comm_vectors = comm_vectors.reshape(1, *comm_vectors.shape)
        else:
            batch_input = True
            batch_agent_positions, batch_r_dests, batch_fuel_levels = self.marine_decode_agents_position_destination_fuel(comm_vectors)
            if self.args.schedule_rewards:
                pairings = self.marine_get_schedules_from_comm_vectors(comm_vectors, batch_agent_positions, batch_r_dests, batch_fuel_levels) # [b, num_agents]
            batch_size = comm_vectors.shape[0]

        BASE = self.args.dim ** 2
        binary_BASE = BASE.bit_length()  # binarized value
        max_fuel = int(self.args.dim)
        binary_max_fuel = max_fuel.bit_length()
        x_per_stat = np.zeros((batch_size, self.args.num_P, self.num_stat_in_channels, self.args.dim, self.args.dim), dtype=float)
        x_act_stat = np.zeros((batch_size, self.args.num_A, self.num_stat_in_channels, self.args.dim, self.args.dim), dtype=float)
        x_per_obs = np.zeros((batch_size, self.args.num_P, self.num_obs_in_channels, 2*self.args.vision+1, 2*self.args.vision+1), dtype=float)
        x_act_obs = np.zeros((batch_size, self.args.num_A, self.num_obs_in_channels, 2*self.args.vision+1, 2*self.args.vision+1), dtype=float)

        for b in range(comm_vectors.shape[0]):  # batch

            b_comm_vectors = comm_vectors[b]
            b_agent_positions = batch_agent_positions[b]
            b_dests = batch_r_dests[b]

            d = b_dests[0]
            b_fuel_levels = batch_fuel_levels[b]

            all_agents_obs_feature = np.zeros((self.num_obs_in_channels, self.args.dim, self.args.dim)) # 0 for routing, 1 for logis, 2 for fuel_level, 3 for wave, 4 for scheduling

            # wave map - initialize 1/3
            all_agents_obs_feature[3, :, :] = -1
            all_agents_state_feature = np.zeros((self.args.num_P + self.args.num_A, self.num_stat_in_channels, self.args.dim, self.args.dim), dtype=float) # 0 for agent/dest position, 1 for next step wave map prediction

            for i in range(self.args.num_P + self.args.num_A):
                p = b_agent_positions[i]
                if i < self.args.num_P: # routing agents
                    all_agents_obs_feature[0, p[0], p[1]] += 1
                else: # logistics agents
                    all_agents_obs_feature[1, p[0], p[1]] += 1

                all_agents_state_feature[i, 0, p[0], p[1]] = i+1
                all_agents_state_feature[i, 0, d[0], d[1]] = -1 # support only 1 destination case as of now

                # Encode wave map
                i_around_info = b_comm_vectors[i, 2 * binary_BASE + binary_max_fuel:]
                i_around_info = i_around_info.reshape((2*self.args.vision + 1, 2*self.args.vision+1, -1))

                for x in range(2*self.args.vision + 1):
                    for y in range(2*self.args.vision + 1):
                        i_around_info_xy = i_around_info[x, y]
                        # axis=2 0~self.ntotal-1: existing agents' ids at the grid
                        # axis=2 self.ntotal: either outside or not
                        # axis=2 self.ntotal+1: either dest or not
                        # axis=2 self.ntotal+2~self.ntotal+3: intensity
                        if i_around_info_xy[self.args.num_P + self.args.num_A] == 1: # outside
                            continue # outside of the grid
                        else:
                            # print('i_around_info_xy', x, y, i_around_info_xy)
                            if (i_around_info_xy[-2] == 0) and (i_around_info_xy[-1] == 0):
                                wave = 0.0
                                pass
                            elif (i_around_info_xy[-2] == 0) and (i_around_info_xy[-1] == 1):
                                wave = float(1/3)
                            elif (i_around_info_xy[-2] == 1) and (i_around_info_xy[-1] == 0):
                                wave = float(2/3)
                            else: # (agent_around_info_xy[-2] == 1) and (agent_around_info_xy[-1] == 1)
                                wave = 1.0
                            assert (p[0] + x - self.args.vision >= 0)
                            assert (p[1] + y - self.args.vision >= 0)
                            all_agents_obs_feature[3, p[0]+x-self.args.vision, p[1]+y-self.args.vision] = wave


            all_agents_obs_feature_pad = np.pad(all_agents_obs_feature, pad_width=((0,0),(self.args.vision,self.args.vision),(self.args.vision,self.args.vision)), constant_values=-1)

            for i in range(self.args.num_P + self.args.num_A):
                p = b_agent_positions[i]
                x_start = p[0]  # + 1 - self.vision
                x_end = p[0] + 2 * self.args.vision + 1  # + self.vision + 1 + 1
                y_start = p[1]  # + 1 - self.vision
                y_end = p[1] + 2 * self.args.vision + 1  # 1 + self.vision + 1
                i_obs_feature = copy.copy(all_agents_obs_feature_pad[:, x_start:x_end, y_start:y_end])
                # record fuel level
                if (i < self.args.num_P) or self.args.limited_refuel:
                    i_obs_feature[2, self.args.vision, self.args.vision] = b_fuel_levels[i]
                if self.args.schedule_rewards:
                    # record pairings
                    i_obs_feature[4, self.args.vision, self.args.vision] = pairings[b, i]

                if i < self.args.num_P: # routing agents
                    x_per_stat[b, i] = all_agents_state_feature[i]
                    x_per_obs[b, i] = i_obs_feature
                else: # logistics agents
                    x_act_stat[b, i-self.args.num_P] = all_agents_state_feature[i]
                    x_act_obs[b, i-self.args.num_P] = i_obs_feature

        if not batch_input:
            x_per_stat = x_per_stat.reshape(*x_per_stat.shape[1:])
            x_act_stat = x_act_stat.reshape(*x_act_stat.shape[1:])
            x_per_obs = x_per_obs.reshape(*x_per_obs.shape[1:])
            x_act_obs = x_act_obs.reshape(*x_act_obs.shape[1:])

        return torch.tensor(x_per_stat), torch.tensor(x_act_stat), torch.tensor(x_per_obs), torch.tensor(x_act_obs)

    
class GAT_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(GAT_Critic, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart

        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)

        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        self.base = base(args, cent_obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            if args.use_LSTM:
                # self.rnn = LSTMLayer(args, self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
                self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            else:
                self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.
        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            if self.args.use_LSTM:
                critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
            else:
                critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks) #


        values = self.v_out(critic_features)

        return values, rnn_states
