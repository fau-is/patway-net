import torch
import os
import matplotlib.pyplot as plt
import src.data as data
import numpy as np

repetition = 0

model = torch.load(os.path.join("../model", f"model_{repetition}"))
interactions_seq = model.get_number_interactions_seq()
number_interactions_seq = len(interactions_seq)

x_seqs, x_statics, y, x_time_vals_final, seq_features, static_features = data.get_sepsis_data('Admission IC', 100, 3)

"""
# Print seq features (first time step)
for idx in range(0, len(seq_features)):
    x, out = model.plot_feat_seq_effect(idx, -2, 2)
    x = x.detach().numpy().squeeze()
    out = out.detach().numpy()
    plt.plot(x, out)
    plt.title(seq_features[idx])
    plt.show()

# Print seq interaction features (first time step
if number_interactions_seq > 0:
    for idx in range(0, number_interactions_seq):
        X_seq, out = model.plot_feat_seq_effect_inter(idx, -1, 1, -1, 1)
        X_seq = X_seq.detach().numpy().squeeze()
        out = out.detach().numpy()
        plt.imshow(out.reshape(int(np.sqrt(len(X_seq))), int(np.sqrt(len(X_seq)))).transpose())
        plt.title(f"{seq_features[interactions_seq[idx][0]]} x {seq_features[interactions_seq[idx][1]]}")
        plt.show()
"""

# Print stat features
for idx in range(0, len(static_features)):
    x, out = model.plot_feat_stat_effect(idx, -2, 2)
    x = x.detach().numpy().squeeze()
    out = out.detach().numpy()
    plt.plot(x, out)
    plt.title(static_features[idx])
    plt.show()


