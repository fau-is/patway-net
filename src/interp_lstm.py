#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 11:10:35 2022

@author: makraus
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

class NaiveCustomLSTM(nn.Module):
    def __init__(self, input_sz: int, hidden_per_feat_sz: int, interactions: list):
        super().__init__()
        self.input_sz = input_sz
        self.interactions = interactions
        self.input_size = input_sz + 2 * len(interactions)
        self.hidden_per_feat_sz = hidden_per_feat_sz
        self.hidden_size = (input_sz + len(interactions)) * hidden_per_feat_sz
        hidden_sz = self.hidden_size
        
        #i_t
        self.U_i = nn.Parameter(torch.Tensor(self.input_size, hidden_sz))
        self.V_i = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i = nn.Parameter(torch.Tensor(hidden_sz))
        
        #f_t
        self.U_f = nn.Parameter(torch.Tensor(self.input_size, hidden_sz))
        self.V_f = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_f = nn.Parameter(torch.Tensor(hidden_sz))
        
        #c_t
        self.U_c = nn.Parameter(torch.Tensor(self.input_size, hidden_sz))
        self.V_c = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_c = nn.Parameter(torch.Tensor(hidden_sz))
        
        #o_t
        self.U_o = nn.Parameter(torch.Tensor(self.input_size, hidden_sz))
        self.V_o = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_o = nn.Parameter(torch.Tensor(hidden_sz))
        
        self.init_weights()
        
        #Create masking to avoid interactions between features
        self.U_mask = torch.zeros(self.U_i.size())
        
        for i in range(input_sz):
            self.U_mask[i, i*hidden_per_feat_sz:(i+1)*hidden_per_feat_sz] = 1
        
        for i in range(0, len(interactions), 2):
            self.U_mask[i + input_sz, i*hidden_per_feat_sz:(i+1)*hidden_per_feat_sz] = 1
            self.U_mask[i + input_sz + 1, i*hidden_per_feat_sz:(i+1)*hidden_per_feat_sz] = 1
     
        
        v_mask_single = torch.ones((hidden_per_feat_sz, hidden_per_feat_sz))
        self.V_mask = torch.block_diag(*[v_mask_single for _ in range(input_sz + len(interactions))]) 
        
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
            
    def forward(self,
                x,
                init_states=None):
        # Assumes x.shape represents (batch_size, sequence_size, input_size)
        
        bs, seq_sz, feat = x.size()
        hidden_seq = []
        feat_seq = list(range(feat)) + list(sum(self.interactions, ()))
        
        if init_states is None:
            h_t, c_t = (
                torch.zeros(bs, self.hidden_size).to(x.device),
                torch.zeros(bs, self.hidden_size).to(x.device),
            )
        else:
            h_t, c_t = init_states
        
        for t in range(seq_sz):
            x_t = x[:, t, :]
            
            i_t = torch.sigmoid(x_t[:,feat_seq] @ (self.U_i * self.U_mask) + h_t @ (self.V_i * self.V_mask) + self.b_i)
            f_t = torch.sigmoid(x_t[:,feat_seq]  @ (self.U_f * self.U_mask) + h_t @ (self.V_f * self.V_mask) + self.b_f)
            g_t = torch.tanh(x_t[:,feat_seq]  @ (self.U_c * self.U_mask) + h_t @ (self.V_c * self.V_mask) + self.b_c)
            o_t = torch.sigmoid(x_t[:,feat_seq]  @ (self.U_o * self.U_mask) + h_t @ (self.V_o * self.V_mask) + self.b_o)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            
            hidden_seq.append(h_t.unsqueeze(0))
        
        #reshape hidden_seq p/ retornar
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)
    
    def single_forward(self, x, feat_id, interaction=False):
        # We start with no history (h_t = 0, c_t = 0)
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        
        if interaction:
            U_i = (self.U_i * self.U_mask)[self.input_sz + 2*feat_id: self.input_sz + 2*feat_id+2, :]
            U_c = (self.U_c * self.U_mask)[self.input_sz + 2*feat_id: self.input_sz + 2*feat_id+2, :]
            U_o = (self.U_o * self.U_mask)[self.input_sz + 2*feat_id: self.input_sz + 2*feat_id+2, :]
            
            b_i = self.b_i[(self.input_sz + feat_id) * self.hidden_per_feat_sz: (self.input_sz + feat_id + 1) * self.hidden_per_feat_sz]
            b_c = self.b_c[(self.input_sz + feat_id) * self.hidden_per_feat_sz: (self.input_sz + feat_id + 1) * self.hidden_per_feat_sz]
            b_o = self.b_o[(self.input_sz + feat_id) * self.hidden_per_feat_sz: (self.input_sz + feat_id + 1) * self.hidden_per_feat_sz]
        
        else:
            U_i = (self.U_i * self.U_mask)[feat_id, :].unsqueeze(0)
            U_c = (self.U_c * self.U_mask)[feat_id, :].unsqueeze(0)
            U_o = (self.U_o * self.U_mask)[feat_id, :].unsqueeze(0)
            
            b_i = self.b_i[feat_id * self.hidden_per_feat_sz: (feat_id + 1) * self.hidden_per_feat_sz]
            b_c = self.b_c[feat_id * self.hidden_per_feat_sz: (feat_id + 1) * self.hidden_per_feat_sz]
            b_o = self.b_o[feat_id * self.hidden_per_feat_sz: (feat_id + 1) * self.hidden_per_feat_sz]
        
        for t in range(seq_sz):
            x_t = x[:, t]    
            i_t = torch.sigmoid((x_t @ U_i)[:,feat_id*self.hidden_per_feat_sz:(feat_id+1)*self.hidden_per_feat_sz] + b_i)
            g_t = torch.tanh((x_t @ U_c)[:,feat_id*self.hidden_per_feat_sz:(feat_id+1)*self.hidden_per_feat_sz] + b_c)
            o_t = torch.sigmoid((x_t @ U_o)[:,feat_id*self.hidden_per_feat_sz:(feat_id+1)*self.hidden_per_feat_sz] + b_o)
            c_t = i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            
            hidden_seq.append(h_t.unsqueeze(0))
            
        #reshape hidden_seq p/ retornar
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)
    
class Net(nn.Module):
    def __init__(self, input_sz: int, hidden_per_feat_sz: int, output_sz: int,
                 interactions: list = []):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_per_feat_sz = hidden_per_feat_sz
        self.lstm = NaiveCustomLSTM(input_sz, hidden_per_feat_sz, interactions)
        self.output_coef = nn.Parameter(torch.randn(self.lstm.hidden_size, output_sz))
        self.output_bias = nn.Parameter(torch.randn(output_sz))
        self.interactions = interactions
    
    def forward(self, x):
        hidden_seq, (h_t, c_t) = self.lstm(x)
        out = h_t @ self.output_coef + self.output_bias
        return out
    
    def plot_effect(self, feat_id, min_v, max_v):
        x = torch.linspace(min_v, max_v).reshape(-1,1,1)
        
        hidden_seq, (h_t, c_t) = self.lstm.single_forward(x, feat_id)
        out = h_t @ self.output_coef[feat_id * self.hidden_per_feat_sz: (feat_id + 1) * self.hidden_per_feat_sz]
        
        return x, out
    
    def plot_effect_inter(self, inter_id, min_f1, max_f1, min_f2, max_f2):
        X = [[x1, x2] for x1 in np.linspace(min_f1, max_f1) for x2 in np.linspace(max_f2, min_f2)]
        X = torch.Tensor(X).unsqueeze(1)
        hidden_seq, (h_t, c_t) = self.lstm.single_forward(X, inter_id, interaction=True)
        
        out = h_t @ self.output_coef[(self.input_sz + inter_id) * self.hidden_per_feat_sz: (self.input_sz + inter_id + 1) * self.hidden_per_feat_sz]
        
        return X, out
    
n = 1000

X = torch.randn(n, 10, 3)
y = (X[:,:, 0]**2).sum(dim=1) + X[:,:,1].sum(dim=1) + (X[:,:,2]**3).sum(dim=1)
# y = X[:,:,0].sum(dim=1) * X[:,:,1].sum(dim=1)
y = (y - y.mean()) / y.std()
y = y.unsqueeze(1)

m = Net(input_sz = 3, 
        hidden_per_feat_sz = 10, 
        output_sz = 1,
        interactions = [(0,1)])

criterion = nn.MSELoss()
optimizer = optim.Adam(m.parameters(), lr=3e-3)

idx = np.arange(n)
for _ in range(10):
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    loss_all = 0
    for i in range(X.shape[0] // 64):
        out = m(X[i*64:(i+1)*64])
        loss = criterion(out, y[i*64:(i+1)*64])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_all += float(loss)
    print(loss_all)

"""
x, out = m.plot_effect(0, -2, 2)
x = x.detach().numpy().squeeze()
out = out.detach().numpy()
plt.plot(x, out)
plt.show()

x, out = m.plot_effect(1, -2, 2)
x = x.detach().numpy().squeeze()
out = out.detach().numpy()
plt.plot(x, out)
plt.show()

x, out = m.plot_effect(2, -2, 2)
x = x.detach().numpy().squeeze()
out = out.detach().numpy()
plt.plot(x, out)
plt.show()
"""

X, out = m.plot_effect_inter(0, -1, 1, -1, 1)
X = X.detach().numpy().squeeze()
out = out.detach().numpy()
plt.imshow(out.reshape(int(np.sqrt(len(X))), int(np.sqrt(len(X)))).transpose())
plt.show()
