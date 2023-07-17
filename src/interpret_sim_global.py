from src.data import get_sim_data
import torch
import numpy as np
import os
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
model = torch.load(os.path.join("../model", f"model_sim"))
x_seqs, x_statics, y, _, seq_features, static_features = get_sim_data('Label', 'Simulation_data_1000.csv')

# Create dataset without prefixes
x_seq_final = np.zeros((len(x_seqs), 12, len(x_seqs[0][0])))
x_stat_final = np.zeros((len(x_seqs), len(x_statics[0])))
for i, x in enumerate(x_seqs):
    x_seq_final[i, :len(x), :] = np.array(x)
    x_stat_final[i, :] = np.array(x_statics[i])

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

time = 11

# assumption: last time step
for t in range(time, time + 1):
    for idx, value in enumerate(seq_features):

        x, out, h_t, out_coef = model.plot_feat_seq_effect(idx, torch.from_numpy(
            x_seq_final[:, t, idx].reshape(-1, 1, 1)).float())
        x = x.detach().numpy().squeeze()
        out = out.detach().numpy()
        out = np.ravel(out)
        out = out.tolist()

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

# plt.rcParams.update({'font.size': 11})
plt.figure(figsize=(12, 4))
plt.rc('font', size=10)
plt.rc('axes', titlesize=13)
plt.rc('axes', labelsize=10)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

y_pos = np.arange(len(feat_names_sorted))
plot = plt.barh(y_pos, feat_imports_sorted, color='lightblue')
# plt.yticks(y_pos, feat_names_sorted)
plt.tick_params(left=False, labelleft=False)
plt.xticks(np.arange(0, 0.21, step=0.05))


def autolabel(plot):
    for idx, rect in enumerate(plot):
        plt.text(0.005, idx - 0.2, feat_names_sorted[idx], color='black')


autolabel(plot)

plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.title("Global feature importance ($t_{%s}$)" % str(time + 1))
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig(f'../plots/simulation/sim_global_feature_importance.pdf', dpi=100)
plt.close(fig1)
