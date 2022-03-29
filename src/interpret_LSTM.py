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


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output


class NaiveCustomLSTM(nn.Module):
    def __init__(self, input_sz: int, hidden_per_feat_sz: int, masking: bool, interactions: list):
        super().__init__()
        self.input_sz = input_sz
        self.masking = masking
        self.interactions = interactions
        self.input_size = input_sz + 2 * len(interactions)
        self.hidden_per_feat_sz = hidden_per_feat_sz
        self.hidden_size = (input_sz + len(interactions)) * hidden_per_feat_sz
        hidden_sz = self.hidden_size

        # i_t
        self.U_i = nn.Parameter(torch.Tensor(self.input_size, hidden_sz))
        self.V_i = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i = nn.Parameter(torch.Tensor(hidden_sz))

        # f_t
        self.U_f = nn.Parameter(torch.Tensor(self.input_size, hidden_sz))
        self.V_f = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_f = nn.Parameter(torch.Tensor(hidden_sz))

        # c_t
        self.U_c = nn.Parameter(torch.Tensor(self.input_size, hidden_sz))
        self.V_c = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_c = nn.Parameter(torch.Tensor(hidden_sz))

        # o_t
        self.U_o = nn.Parameter(torch.Tensor(self.input_size, hidden_sz))
        self.V_o = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_o = nn.Parameter(torch.Tensor(hidden_sz))

        self.init_weights()

        if self.masking:
            # Create masking to avoid interactions between features
            self.U_mask = torch.zeros(self.U_i.size())

            for i in range(input_sz):
                self.U_mask[i, i * hidden_per_feat_sz:(i + 1) * hidden_per_feat_sz] = 1

            for i in range(0, len(interactions), 2):
                self.U_mask[i + input_sz, i * hidden_per_feat_sz:(i + 1) * hidden_per_feat_sz] = 1
                self.U_mask[i + input_sz + 1, i * hidden_per_feat_sz:(i + 1) * hidden_per_feat_sz] = 1

            v_mask_single = torch.ones((hidden_per_feat_sz, hidden_per_feat_sz))
            self.V_mask = torch.block_diag(*[v_mask_single for _ in range(input_sz + len(interactions))])

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, init_states=None):

        # Assumes x.shape represents (batch_size, sequence_size, input_size)
        bs, seq_sz, feat = x.size()
        hidden_seq = []

        if self.masking:
            feat_seq = list(range(feat)) + list(sum(self.interactions, ()))  # concat

        if init_states is None:
            h_t, c_t = (
                torch.zeros(bs, self.hidden_size).to(x.device),
                torch.zeros(bs, self.hidden_size).to(x.device),
            )
        else:
            h_t, c_t = init_states

        for t in range(seq_sz):
            x_t = x[:, t, :]

            if self.masking:
                i_t = torch.sigmoid(
                    x_t[:, feat_seq] @ (self.U_i * self.U_mask) + h_t @ (self.V_i * self.V_mask) + self.b_i)
                f_t = torch.sigmoid(
                    x_t[:, feat_seq] @ (self.U_f * self.U_mask) + h_t @ (self.V_f * self.V_mask) + self.b_f)
                g_t = torch.tanh(
                    x_t[:, feat_seq] @ (self.U_c * self.U_mask) + h_t @ (self.V_c * self.V_mask) + self.b_c)
                o_t = torch.sigmoid(
                    x_t[:, feat_seq] @ (self.U_o * self.U_mask) + h_t @ (self.V_o * self.V_mask) + self.b_o)
                c_t = f_t * c_t + i_t * g_t
                h_t = o_t * torch.tanh(c_t)
            else:
                i_t = torch.sigmoid(x_t @ self.U_i + h_t @ self.V_i + self.b_i)
                f_t = torch.sigmoid(x_t @ self.U_f + h_t @ self.V_f + self.b_f)
                g_t = torch.tanh(x_t @ self.U_c + h_t @ self.V_c + self.b_c)
                o_t = torch.sigmoid(x_t @ self.U_o + h_t @ self.V_o + self.b_o)
                c_t = f_t * c_t + i_t * g_t
                h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))

        # reshape hidden_seq p/ retornar
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)

    def single_forward(self, x, feat_id, interaction=False):
        # We start with no history (h_t = 0, c_t = 0)
        bs, seq_sz, _ = x.size()
        hidden_seq = []

        # get interaction/feature specific parameters
        if interaction:
            U_i = (self.U_i * self.U_mask)[self.input_sz + 2 * feat_id: self.input_sz + 2 * feat_id + 2, :]
            # U_f = (self.U_f * self.U_mask)[self.input_sz + 2 * feat_id: self.input_sz + 2 * feat_id + 2, :]
            U_c = (self.U_c * self.U_mask)[self.input_sz + 2 * feat_id: self.input_sz + 2 * feat_id + 2, :]
            U_o = (self.U_o * self.U_mask)[self.input_sz + 2 * feat_id: self.input_sz + 2 * feat_id + 2, :]

            b_i = self.b_i[(self.input_sz + feat_id) * self.hidden_per_feat_sz: (
                                                            self.input_sz + feat_id + 1) * self.hidden_per_feat_sz]
            # b_f = self.b_f[(self.input_sz + feat_id) * self.hidden_per_feat_sz: (self.input_sz + feat_id + 1) * self.hidden_per_feat_sz]
            b_c = self.b_c[(self.input_sz + feat_id) * self.hidden_per_feat_sz: (
                                                            self.input_sz + feat_id + 1) * self.hidden_per_feat_sz]
            b_o = self.b_o[(self.input_sz + feat_id) * self.hidden_per_feat_sz: (
                                                            self.input_sz + feat_id + 1) * self.hidden_per_feat_sz]

        else:
            U_i = (self.U_i * self.U_mask)[feat_id, :].unsqueeze(0)
            # U_f = (self.U_f * self.U_mask)[feat_id, :].unsqueeze(0)
            U_c = (self.U_c * self.U_mask)[feat_id, :].unsqueeze(0)
            U_o = (self.U_o * self.U_mask)[feat_id, :].unsqueeze(0)

            b_i = self.b_i[feat_id * self.hidden_per_feat_sz: (feat_id + 1) * self.hidden_per_feat_sz]
            # b_f = self.b_f[feat_id * self.hidden_per_feat_sz: (feat_id + 1) * self.hidden_per_feat_sz]
            b_c = self.b_c[feat_id * self.hidden_per_feat_sz: (feat_id + 1) * self.hidden_per_feat_sz]
            b_o = self.b_o[feat_id * self.hidden_per_feat_sz: (feat_id + 1) * self.hidden_per_feat_sz]

        for t in range(seq_sz):
            x_t = x[:, t]
            i_t = torch.sigmoid((x_t @ U_i)[:, feat_id * self.hidden_per_feat_sz: (feat_id + 1) * self.hidden_per_feat_sz] + b_i)
            # f_t = torch.sigmoid((x_t @ U_f)[:, feat_id * self.hidden_per_feat_sz:(feat_id + 1) * self.hidden_per_feat_sz] + b_f)
            g_t = torch.tanh((x_t @ U_c)[:, feat_id * self.hidden_per_feat_sz:(feat_id + 1) * self.hidden_per_feat_sz] + b_c)
            o_t = torch.sigmoid((x_t @ U_o)[:, feat_id * self.hidden_per_feat_sz:(feat_id + 1) * self.hidden_per_feat_sz] + b_o)
            # todo: f_t not used?
            c_t = i_t * g_t  # * f_t
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))

        # reshape hidden_seq p/ retornar
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)


class Net(nn.Module):
    def __init__(self, input_sz_seq: int, hidden_per_seq_feat_sz: int, input_sz_stat: int, output_sz: int,
                 interactions_seq_itr: int, interactions_seq_best: int, interactions_seq_auto: bool,
                 x_seq: torch.Tensor, masking: bool, x_stat: torch.Tensor, y: torch.Tensor,
                 interactions_seq: list = []):
        """
        :param input_sz_seq:
        :param hidden_per_seq_feat_sz:
        :param input_sz_stat: assumption, each feature is represented by a single feature; not interactions?
        :param output_sz:
        :param interactions_seq:
        """

        super().__init__()
        self.input_sz_seq = input_sz_seq
        self.hidden_per_feat_sz = hidden_per_seq_feat_sz
        self.interactions_seq = interactions_seq
        self.interactions_seq_best = interactions_seq_best
        self.interactions_seq_auto = interactions_seq_auto
        self.interactions_seq_itr = interactions_seq_itr
        self.masking = masking

        if self.interactions_seq_auto and self.interactions_seq == [] and self.masking:
            self.interactions_seq = self.get_interactions_seq_auto(x_seq, y)

        self.lstm = NaiveCustomLSTM(input_sz_seq, hidden_per_seq_feat_sz, self.masking, self.interactions_seq)
        self.mlps = nn.ModuleList([MLP(1, 10) for i in range(input_sz_stat)])
        self.output_coef = nn.Parameter(torch.randn(self.lstm.hidden_size + input_sz_stat, output_sz))
        self.output_bias = nn.Parameter(torch.randn(output_sz))
        self.input_sz_stat = input_sz_stat

    def get_number_interactions_seq(self):
        return self.interactions_seq

    def get_interactions_seq_auto(self, x_seq, y):
        """
        Determines interactions automatically from data
        :param x_seq:
        :return:
        """

        import random
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score

        x_seq_features = list(range(x_seq.shape[2]))

        num_iters = self.interactions_seq_itr
        num_best_inters = self.interactions_seq_best
        results = pd.DataFrame({'Pair': [], 'Measure': []})

        for _ in range(0, num_iters):

            print(f"Iteration: {_}")

            # Get feature pair
            rnd_feat_source = random.choice(x_seq_features)
            rnd_feat_target = random.choice(x_seq_features)
            feat_pair = [rnd_feat_target, rnd_feat_source]
            feat_pair.sort()

            # Exists interaction already or is the interaction a loop?
            if str(feat_pair) in results["Pair"].values or rnd_feat_source == rnd_feat_target:
                continue

            # Get data
            x_seq_sample = torch.cat([x_seq[:, :, feat_pair[0]].reshape(-1, x_seq.shape[1], 1),
                                      x_seq[:, :, feat_pair[1]].reshape(-1, x_seq.shape[1], 1)], dim=2)

            # iterate over time
            measure = 0
            max_measure = 0

            # Learn and apply linear model
            x_seq_sample_train, x_seq_sample_test, y_train, y_test = train_test_split(x_seq_sample[:, :, :].reshape(-1,
                                                                                                                    x_seq_sample.shape[
                                                                                                                        1] *
                                                                                                                    x_seq_sample.shape[
                                                                                                                        2]),
                                                                                      y,
                                                                                      train_size=0.7, shuffle=False)
            c = [pow(10, -3), pow(10, -2), pow(10, -1), pow(10, 0), pow(10, 1), pow(10, 2), pow(10, 3)]

            for param in c:
                model = LogisticRegression(penalty='l2', solver='lbfgs', C=param)
                model.fit(x_seq_sample_train, np.ravel(y_train))
                preds_proba = model.predict_proba(x_seq_sample_test)
                preds_proba = [pred_proba[1] for pred_proba in preds_proba]

                try:
                    measure = roc_auc_score(y_true=y_test, y_score=preds_proba)
                    if measure > max_measure:
                        max_measure = measure
                except:
                    pass

            # Save result
            results = results.append({'Pair': str(feat_pair), 'Measure': max_measure}, ignore_index=True)

        # Retrieve best interactions
        results = results.nlargest(n=num_best_inters, columns=['Measure'])

        if results.empty:
            print("No interactions found!!!")
            return []
        else:
            return [tuple(eval(x)) for x in results["Pair"].values]

    def forward(self, x_seq, x_stat):

        hidden_seq, (h_t, c_t) = self.lstm(x_seq)

        out_mlp = []
        for i, mlp in enumerate(self.mlps):
            if i == 0:
                out_mlp = mlp(x_stat[:, i].reshape(-1, 1).float())
            else:
                out_mlp_temp = mlp(x_stat[:, i].reshape(-1, 1).float())
                out_mlp = torch.cat((out_mlp, out_mlp_temp), dim=1)

        out = torch.cat((h_t, out_mlp), dim=1) @ self.output_coef.float() + self.output_bias

        """
        output_sz = 1
        self.output_coef = nn.Parameter(torch.randn(self.input_sz_stat, output_sz))
        out = x_stat @ self.output_coef.double() + self.output_bias
        """

        return out

    def plot_feat_seq_effect_custom(self, feat_id, min_v, max_v):
        x = torch.linspace(min_v, max_v).reshape(-1, 1, 1)

        hidden_seq, (h_t, c_t) = self.lstm.single_forward(x, feat_id)
        out = h_t @ self.output_coef[feat_id * self.hidden_per_feat_sz: (feat_id + 1) * self.hidden_per_feat_sz]

        return x, out

    def plot_feat_seq_effect(self, feat_id, x):

        hidden_seq, (h_t, c_t) = self.lstm.single_forward(x, feat_id)
        out = h_t @ self.output_coef[feat_id * self.hidden_per_feat_sz: (feat_id + 1) * self.hidden_per_feat_sz]

        return x, out

    def plot_feat_seq_effect_inter_custom(self, inter_id, min_f1, max_f1, min_f2, max_f2):
        x = [[x1, x2] for x1 in np.linspace(min_f1, max_f1) for x2 in np.linspace(max_f2, min_f2)]
        x = torch.Tensor(x).unsqueeze(1)
        hidden_seq, (h_t, c_t) = self.lstm.single_forward(x, inter_id, interaction=True)

        out = h_t @ self.output_coef[(self.input_sz_seq + inter_id) * self.hidden_per_feat_sz: (
                                                    self.input_sz_seq + inter_id + 1) * self.hidden_per_feat_sz]
        return x, out

    def plot_feat_seq_effect_inter(self, inter_id, x):
        hidden_seq, (h_t, c_t) = self.lstm.single_forward(x, inter_id, interaction=True)

        out = h_t @ self.output_coef[(self.input_sz_seq + inter_id) * self.hidden_per_feat_sz: (
                                                    self.input_sz_seq + inter_id + 1) * self.hidden_per_feat_sz]
        return x, out

    def plot_feat_stat_effect_custom(self, feat_id, min_v, max_v):
        x = torch.linspace(min_v, max_v).reshape(-1, 1)
        mlp = self.mlps[feat_id]
        mlp_out = mlp(x)
        out = self.output_coef[feat_id + self.lstm.hidden_size: (feat_id + 1) + self.lstm.hidden_size] * mlp_out

        return x, out

    def plot_feat_stat_effect(self, feat_id, x):
        mlp = self.mlps[feat_id]
        mlp_out = mlp(x)
        out = self.output_coef[feat_id + self.lstm.hidden_size: (feat_id + 1) + self.lstm.hidden_size] * mlp_out

        return x, out