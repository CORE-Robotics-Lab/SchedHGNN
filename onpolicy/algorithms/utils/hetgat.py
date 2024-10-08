from tkinter.messagebox import NO
from turtle import forward
import torch.nn as nn
import torch
from onpolicy.algorithms.hetgat_mappo.graphs.fastreal import MultiHeteroGATLayerReal
"""HetGAT module"""

class HetGATLayer(nn.Module):
    def __init__(self, args, outputs_dim, device = torch.device('cpu'), input_size=None):
        super(HetGATLayer, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_heads
        self.device = device
        self.args = args
        self.outputs_dim = outputs_dim
        self.n_types = self.args.n_types # implement a better to deal with different number of agent types
        self.action_have_vision = 'Marine' in self.args.env_name
        if self.n_types == 2:
            state_len = 2 * self.args.num_P + 2 * self.args.num_A + 1
            if 'Marine' in self.args.env_name:
                state_len += 2 * self.args.num_dest
                state_len += self.args.num_P
                if self.args.limited_refuel:
                    state_len += self.args.num_A

            in_dim = {'P':input_size, 'A':input_size, 'X':input_size, 'state':state_len}
            hid_dim = {'P':self.hidden_size, 'A':self.hidden_size, 'state':self.hidden_size}
            hid_dim_input = {'P':self.hidden_size*self.num_heads,
                            'A':self.hidden_size*self.num_heads,
                            'state':self.hidden_size*self.num_heads}
            out_dim = {'P':self.hidden_size, 'A':self.hidden_size, 'state':state_len}

            if self.args.tensor_obs:
                in_dim['P'] = 128 + 16 # input dimension of perception agents is equal to the sum of state dim and obs dim
                # in_dim['A'] = 128 # dimension
                if self.action_have_vision:  # Ocean Env
                    in_dim['A'] = 128 + 16 # action agents also have vision
                else: # PCP Env, FC Env
                    in_dim['A'] = 128


        else:
            raise NotImplementedError

        self.hetgat_layer1 = MultiHeteroGATLayerReal(in_dim, hid_dim, self.hetnet_num_heads, device=self.device,
                                                     action_have_vision=self.action_have_vision)
        self.hetgat_layer2 = MultiHeteroGATLayerReal(hid_dim_input, out_dim, self.hetnet_num_heads, device=self.device,
                                                     action_have_vision=self.action_have_vision, merge='avg')

        self.norm = nn.LayerNorm(outputs_dim)

    def forward(self, feat_dict, cent_obs, graph_batch, batch_size):

        if cent_obs != None:
            feat_dict['state'] = cent_obs
        h1 = self.hetgat_layer1(graph_batch.to(self.device), feat_dict)
        q_out= self.hetgat_layer2(graph_batch.to(self.device), h1, last_layer=True)
        
        if self.args.num_A == 0 and self.args.num_X == 0:
            x = q_out['P'].view(batch_size,self.args.num_P,-1)
        elif self.args.num_X == 0:
            x = torch.cat((q_out['P'].view(batch_size,self.args.num_P,-1), q_out['A'].view(batch_size,self.args.num_A,-1)), dim=1)
        # x = self.norm(q_out)
        if cent_obs is not None:
            state = q_out['state']
        else:
            state = None
        return x, state
