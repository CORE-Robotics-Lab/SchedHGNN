from turtle import forward
import torch
import torch.nn as nn

"""RNN modules."""


class RNNLayer(nn.Module):
    def __init__(self, inputs_dim, outputs_dim, recurrent_N, use_orthogonal):
        super(RNNLayer, self).__init__()
        self._recurrent_N = recurrent_N
        self._use_orthogonal = use_orthogonal

        self.rnn = nn.GRU(inputs_dim, outputs_dim, num_layers=self._recurrent_N)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        self.norm = nn.LayerNorm(outputs_dim)

    def forward(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.rnn(x.unsqueeze(0),
                              (hxs * masks.repeat(1, self._recurrent_N).unsqueeze(-1)).transpose(0, 1).contiguous())
            x = x.squeeze(0)
            hxs = hxs.transpose(0, 1)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            has_zeros = ((masks[1:] == 0.0)
                         .any(dim=-1)
                         .nonzero()
                         .squeeze()
                         .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.transpose(0, 1)

            outputs = []
            for i in range(len(has_zeros) - 1):
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                temp = (hxs * masks[start_idx].view(1, -1, 1).repeat(self._recurrent_N, 1, 1)).contiguous()
                rnn_scores, hxs = self.rnn(x[start_idx:end_idx], temp)
                outputs.append(rnn_scores)

            x = torch.cat(outputs, dim=0)

            # flatten
            x = x.reshape(T * N, -1)
            hxs = hxs.transpose(0, 1)

        x = self.norm(x)
        return x, hxs


class LSTMLayer(nn.Module):
    def __init__(self, args, inputs_dim, outputs_dim, recurrent_N, use_orthogonal):
        super(LSTMLayer, self).__init__()
        self._recurrent_N = recurrent_N
        self._use_orthogonal = use_orthogonal
        
        self.lstm = nn.LSTMCell(inputs_dim, outputs_dim)
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        self.norm = nn.LayerNorm(outputs_dim)
        self.args = args
        
    def forward(self, x, h):
        h = h.squeeze()
        h0, c0 = torch.split(h, [self.args.num_P+self.args.num_A, self.args.num_P+self.args.num_A], dim=-2)
        xshape = x.shape
        hidden_size = xshape[-1]
        if len(xshape) > 2:
            batch_input = True
            batch_size = xshape[0]
            x = x.reshape((-1, hidden_size))
            h0 = h0.reshape((-1, hidden_size))
            c0 = c0.reshape((-1, hidden_size))
        else:
            batch_input = False
        x, h = self.lstm(x, (h0,c0))
        h = torch.cat((x, h))
        h = h.unsqueeze(dim=1)
        if batch_input:
            x = x.reshape((batch_size, -1, hidden_size))
            h = h.reshape((batch_size, -1, 1, hidden_size))
        return x, h


class LSTMLayer1(nn.Module):
    def __init__(self, args, inputs_dim, outputs_dim, recurrent_N, use_orthogonal):
        super(LSTMLayer1, self).__init__()
        self._recurrent_N = recurrent_N
        self._use_orthogonal = use_orthogonal

        self.lstm = nn.LSTMCell(inputs_dim, outputs_dim)
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        self.norm = nn.LayerNorm(outputs_dim)
        self.args = args

    def forward(self, x, h):
        h = h.squeeze()
        h0, c0 = torch.split(h, [self.args.num_P + self.args.num_A, self.args.num_P + self.args.num_A], dim=-2)
        if len(x) > 2:
            x = x.reshape(-1, x.shape[-1])
            h0 = h0.reshape(-1, h0.shape[-1])
            c0 = c0.reshape(-1, c0.shape[-1])
        x, h = self.lstm(x, (h0, c0))
        h = torch.cat((x, h))
        h = h.unsqueeze(dim=1)
        return x, h
