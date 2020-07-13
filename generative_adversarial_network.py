import torch
import torch.nn as nn

class generator(nn.Module):
    def __init__(self, hidden_dim, z_dim, val_dim, n_vars):
        super(generator, self).__init__()
        self.lstm = nn.LSTMCell(input_size=z_dim, hidden_size=hidden_dim)
        self.lin = nn.Linear(hidden_dim, val_dim)

        self.n_vars = n_vars
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.val_dim = val_dim

    def forward(self, input):
        k, batchnum, input_dim = input.size()

        assert k == self.n_vars
        assert input_dim == self.z_dim

        output = torch.zeros((0, batchnum, self.val_dim), dtype=torch.float)

        h = torch.zeros(batchnum, self.hidden_dim)
        c = torch.zeros(batchnum, self.hidden_dim)

        if next(self.parameters()).is_cuda:
            h = h.cuda()
            c = c.cuda()
            output = output.cuda()

        for i in range(self.n_vars):
            h, c = self.lstm(input[i], (h, c))
            oi = self.lin(h).abs()
            oi = torch.unsqueeze(oi, 0)
            output = torch.cat((output, oi), 0)

        return output


class discriminator(nn.Module):
    def __init__(self, hidden_dim, val_dim):
        super(discriminator, self).__init__()
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

