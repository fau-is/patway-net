import copy
import math
import numpy as np
import torch
import torch.nn as nn
import warnings


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
        output = self.fc3(act1)
        output = self.act3(output)
        return output


class SLP(nn.Module):
    def __init__(self, input_size):
        super(SLP, self).__init__()
        self.input_size = input_size
        self.act1 = torch.nn.ReLU()
        self.fc = torch.nn.Linear(self.input_size, 1)
        self.act2 = torch.nn.Sigmoid()

    def forward(self, x):
        act1 = self.act1(x)
        output = self.fc(act1)
        output = self.act2(output)
        return output
