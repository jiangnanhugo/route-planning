import torch
import torch.nn as nn

class SeqV2(nn.Module):
    def __init__(self, hidden_dim, z_dim, val_dim, n_vars):
        super(SeqV2, self).__init__()
        self.lstm = nn.LSTMCell(input_size=z_dim, hidden_size=hidden_dim)
        self.lin = nn.Linear(hidden_dim, val_dim)

        self.n_vars = n_vars
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.val_dim = val_dim

    def forward(self, z):
        k, batchnum, input_dim = z.size()

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
            h, c = self.lstm(z[i], (h, c))
            out = self.lin(h)
            out = out.abs()

            out = torch.unsqueeze(out, 0)
            output = torch.cat((output, out), 0)

        return output
