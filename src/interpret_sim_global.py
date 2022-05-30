from src.data import get_sim_data
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.preprocessing import MinMaxScaler

# https://captum.ai/api/feature_permutation.html
from captum.attr import FeaturePermutation, ShapleyValueSampling

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

model = torch.load(os.path.join("../model", f"model_sim"))

x_seqs, x_statics, y, _, seq_features, static_features = get_sim_data('Label', 'Simulation_data_1000.csv')

# Create dataset without prefixes
x_seq_final = np.zeros((len(x_seqs), 12, len(x_seqs[0][0])))
x_stat_final = np.zeros((len(x_seqs), len(x_statics[0])))
for i, x in enumerate(x_seqs):
    x_seq_final[i, :len(x), :] = np.array(x)
    x_stat_final[i, :] = np.array(x_statics[i])


# todo: static features + sequential (last time step)
# outcome: bar plot; x axis = feature name; y axis = importance value
# permutation -> areas
# no interactions


feat_imports = []
feat_names = []

for idx, value in enumerate(static_features):

    x, out = model.plot_feat_stat_effect(idx, torch.from_numpy(x_stat_final[:, idx].reshape(-1, 1)).float())
    x = x.detach().numpy().squeeze()
    out = out.detach().numpy()
    out = np.ravel(out)

    out_min = min(out)
    out_delta = 0 - out_min
    out = [x + out_delta for x in out]

    if len(set(out)) <= 2:
        feat_imports.append(max(out))
    else:
        from scipy import integrate
        sorted_index = np.argsort(np.array(x))
        out_sorted = np.array(out)[sorted_index]
        x_sorted = np.array(x)[sorted_index]
        feat_imports.append(integrate.trapz(y=out_sorted, x=x_sorted))

    feat_names.append(value)


for t in range(11, 12):
    for idx, value in enumerate(seq_features):

        x, out, h_t, out_coef = model.plot_feat_seq_effect(idx, torch.from_numpy(x_seq_final[:, t, idx].reshape(-1, 1, 1)).float())
        x = x.detach().numpy().squeeze()
        out = out.detach().numpy()
        out = np.ravel(out)

        out_min = min(out)
        out_delta = 0 - out_min
        out = [x + out_delta for x in out]

        if len(set(out)) <= 2:
            feat_imports.append(max(out))
        else:
            from scipy import integrate

            sorted_index = np.argsort(np.array(x))
            out_sorted = np.array(out)[sorted_index]
            x_sorted = np.array(x)[sorted_index]
            feat_imports.append(integrate.trapz(y=out_sorted, x=x_sorted))
        feat_names.append(value)

sorted_index = np.argsort(np.array(feat_imports))
feat_imports_sorted = np.array(feat_imports)[sorted_index]
feat_names_sorted = np.array(feat_names)[sorted_index]
feat_names_sorted = feat_names_sorted.tolist()
feat_imports_sorted = feat_imports_sorted.tolist()

y_pos = np.arange(len(feat_names_sorted))
plt.barh(y_pos, feat_imports_sorted)
plt.yticks(y_pos, feat_names_sorted)
plt.xticks(np.arange(0, 0.21, step=0.05))
plt.show()



"""
# feature_perm = FeaturePermutation(model)
feature_perm = ShapleyValueSampling(model)

attr = feature_perm.attribute((torch.from_numpy(x_seq_final).float(),
                               torch.from_numpy(x_stat_final).float()))

# 5 seq and 4 stat features
results = torch.concat((attr[0][:,-1,:], attr[1]), dim=1)
results = results.numpy()
means = []

for idx in range(0, len(seq_features + static_features)):
    means.append(np.mean(results[:,idx]))

importances = pd.DataFrame(
    result.importances[sorted_importances_idx].T,
    columns=X.columns[sorted_importances_idx],
)
ax = importances.plot.box(vert=False, whis=10)
ax.set_title("Permutation Importances (test set)")
ax.axvline(x=0, color="k", linestyle="--")
ax.set_xlabel("Decrease in accuracy score")
ax.figure.tight_layout()
"""
