import torch
import torch.nn as nn


class DRNNV1(nn.Module):
    def __init__(self, hidden_dim, val_dim):
        super(DRNNV1, self).__init__()
        self.rnn = nn.RNN(input_size=val_dim, hidden_size=hidden_dim, num_layers=1)
        self.lin = nn.Linear(hidden_dim, 1)
        self.h0 = torch.randn((1, hidden_dim))

        self.val_dim = val_dim
        self.hidden_dim = hidden_dim

    def forward(self, inp):
        n_vars, n_batch, n_val = inp.size()
        assert n_val == self.val_dim

        h0 = torch.unsqueeze(self.h0.repeat((n_batch, 1)), 0)
        if next(self.parameters()).is_cuda:
            h0 = h0.cuda()

        rnn_oup, rnn_hn = self.rnn(inp, h0)

        return torch.sigmoid(self.lin(rnn_hn))
