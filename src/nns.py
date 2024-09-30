import torch
import torch.nn as nn
import numpy as np

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        self.output_coef = nn.Parameter(torch.randn(self.lstm.hidden_size, 1))
        self.output_bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        hidden_seq, (h_t, c_t) = self.lstm(x)

        out = h_t @ self.output_coef.float() + self.output_bias

        return torch.reshape(out, (-1, 1))


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.act1 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(self.hidden_size, 1)
        self.act3 = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        act1 = self.act1(hidden)
        out = self.fc3(act1)
        out = self.act3(out)
        return out


class SLP(nn.Module):
    def __init__(self, input_size):
        super(SLP, self).__init__()
        self.input_size = input_size
        self.act1 = torch.nn.ReLU()
        self.fc = torch.nn.Linear(self.input_size, 1)
        self.act2 = torch.nn.Sigmoid()

    def forward(self, x):
        act1 = self.act1(x)
        out = self.fc(act1)
        out = self.act2(out)
        return out
