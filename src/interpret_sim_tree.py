from src.data import get_sim_data
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

model = torch.load(os.path.join("../model", f"model_sim"))
x_seqs, x_statics, y, _, seq_features, static_features = get_sim_data('Label', 'Simulation_data_1000.csv')
max_depths = [2, 3, 4]

# Create dataset without prefixes
x_seq_final = np.zeros((len(x_seqs), 12, len(x_seqs[0][0])))
x_stat_final = np.zeros((len(x_seqs), len(x_statics[0])))
for i, x in enumerate(x_seqs):
    x_seq_final[i, :len(x), :] = np.array(x)
    x_stat_final[i, :] = np.array(x_statics[i])

for idx in range(0, len(max_depths)):

    model = DecisionTreeRegressor(max_depth=max_depths[idx])
    model.fit(x_stat_final, np.ravel(y))

    fig = plt.figure(figsize=(40,40))
    plot_tree(model, feature_names=static_features, filled=True)
    plt.show()
    fig.savefig(f"../plots/decision_tree_sim_{max_depths[idx]}.pdf")
    print(static_features)
    print(model.feature_importances_)