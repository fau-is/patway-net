import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import src.data as data
from src.main import time_step_blow_up

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
repetition = 0
# Map location param is required as the model was trained on gpu
model = torch.load(os.path.join("../model", f"model_{repetition}"), map_location=torch.device('cpu'))
interactions_seq = model.get_number_interactions_seq()
number_interactions_seq = len(interactions_seq)

# Note: static features include binary features created from the diagnosis feature or not
x_seqs, x_statics, y, x_time_vals_final, seq_features, static_features = data.get_sepsis_data('Admission IC', 50, 3)
x_seqs_final, x_statics_final, y_final = time_step_blow_up(x_seqs, x_statics, y, 50)

feat_imports = []
feat_names = []

for idx, value in enumerate(static_features):

    x, out = model.plot_feat_stat_effect(idx, torch.from_numpy(x_statics_final[:, idx].reshape(-1, 1)).float())
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

time = 8

# Assumption: most admissions to ic in time step 9
for t in range(time, time + 1):
    for idx, value in enumerate(seq_features):

        if value == 'Leucocytes' or value == 'CRP' or value == 'Release A':
            x, out, h_t, out_coef = model.plot_feat_seq_effect(idx, torch.from_numpy(
                x_seqs_final[:, t, idx].reshape(-1, 1, 1)).float())
            x = x.detach().numpy().squeeze()
            out = out.detach().numpy()
            out = np.ravel(out)
            out = out.tolist()

            out_min = min(out)
            out_delta = 0 - out_min
            out = [o + out_delta for o in out]

            if len(set(out)) <= 2:
                feat_imports.append(max(out))
            else:
                from scipy import integrate

                sorted_index = np.argsort(np.array(x))
                out_sorted = np.array(out)[sorted_index]
                x_sorted = np.array(x)[sorted_index]
                feat_imports.append(integrate.trapz(y=out_sorted, x=x_sorted))
            feat_names.append(value)
    else:
        continue

sorted_index = np.argsort(np.array(feat_imports))
feat_imports_sorted = np.array(feat_imports)[sorted_index]
feat_names_sorted = np.array(feat_names)[sorted_index]
feat_names_sorted = feat_names_sorted.tolist()
feat_imports_sorted = feat_imports_sorted.tolist()

plt.figure(figsize=(5, 6))
plt.rc('axes', titlesize=10)
plt.rc('axes', labelsize=10)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
y_pos = np.arange(len(feat_names_sorted))
plot = plt.barh(y_pos, feat_imports_sorted, color='steelblue')
# plt.yticks(y_pos, feat_names_sorted)
plt.tick_params(left=False, labelleft=False)
plt.xticks(np.arange(0, 4, step=0.5))


def autolabel(plot):
    for idx, rect in enumerate(plot):
        plt.text(0.05, idx - 0.25, feat_names_sorted[idx], color='black')


autolabel(plot)

plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.title("Global feature importance ($t_{%s}$)" % str(time + 1))
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig(f'../plots/sepsis_global_feature_importance.pdf', dpi=100)
plt.close(fig1)