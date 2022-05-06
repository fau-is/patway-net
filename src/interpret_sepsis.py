import torch
import os
import matplotlib.pyplot as plt
import src.data as data
from src.main import time_step_blow_up
import numpy as np

repetition = 0
model = torch.load(os.path.join("../model", f"model_{repetition}"))
interactions_seq = model.get_number_interactions_seq()
number_interactions_seq = len(interactions_seq)

x_seqs, x_statics, y, x_time_vals_final, seq_features, static_features = data.get_sepsis_data('Admission IC', 30, 3)
x_seqs_final, x_statics_final, y_final = time_step_blow_up(x_seqs, x_statics, y, 30)







# Print seq interaction features (first time step
t = 7
if number_interactions_seq > 0:
    for idx in range(0, number_interactions_seq):

        a = torch.from_numpy(x_seqs_final[:, t, interactions_seq[idx][0]].reshape(-1, 1, 1))
        b = torch.from_numpy(x_seqs_final[:, t, interactions_seq[idx][1]].reshape(-1, 1, 1))
        x = torch.cat((a, b), dim=2)

        X_seq, out = model.plot_feat_seq_effect_inter(idx, x)
        X_seq = X_seq.detach().numpy().squeeze()
        out = out.detach().numpy()

        max_size = int(np.sqrt(len(X_seq))) ** 2
        out = out[0:max_size]
        # a_vals = len(set(X_seq[:,0]))
        # b_vals = len(set(X_seq[:,1]))
        # im = plt.imshow(out.reshape(int(np.sqrt(len(X_seq))), int(np.sqrt(len(X_seq)))).transpose())
        # todo:
        im = plt.imshow(out.reshape(int(np.sqrt(len(X_seq))), int(np.sqrt(len(X_seq)))).transpose())
                        # vmin=0, vmax=1)
        cbar = plt.colorbar(im)
        # cbar.set_label("")
        plt.title(f"Interaction:{seq_features[interactions_seq[idx][0]]} x {seq_features[interactions_seq[idx][1]]}")
        plt.xlabel(f"{seq_features[interactions_seq[idx][0]]}")
        plt.ylabel(f"{seq_features[interactions_seq[idx][1]]}")
        plt.show()


