from lime import lime_tabular
import torch
import os
import src.data as data
from src.main import time_step_blow_up
import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Ridge, Lasso

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

name = "4_98"
baseline = "lstm"
model = torch.load(os.path.join("../../model", f"model_{baseline}_{name}"), map_location=torch.device('cpu'))

x_seqs, x_statics, y, x_time_vals_final, seq_features, static_features = data.get_sepsis_data('Admission IC', 50, 3)
x_seqs_final, x_statics_final, y_final = time_step_blow_up(x_seqs, x_statics, y, 50)
case = -1
file_format = "pdf"  # pdf, png

x_seqs_final = torch.from_numpy(x_seqs_final)
x_statics_final = torch.from_numpy(x_statics_final)
y_final = torch.from_numpy(y_final)

x_statics_final_ = torch.reshape(x_statics_final, (-1, 1, x_statics_final.shape[1]))
x_stats = copy.copy(x_statics_final_)
T = x_seqs_final.shape[1]
for t in range(1, T):
    x_stats = torch.concat((x_stats, x_statics_final_), 1)
x_seqs_final = torch.concat((x_seqs_final, x_stats), 2).float()

# x_seqs_final = np.array(x_seqs_final)

explainer = lime_tabular.RecurrentTabularExplainer(x_seqs_final, training_labels=y_final,
                                                   feature_names=seq_features + static_features,
                                                   discretize_continuous=True,
                                                   class_names=['Yes', 'No'])
                                                   # kernel_width=0.01)

exp = explainer.explain_instance(x_seqs_final[case], model, num_features=50*39,
                                 labels=(0,), num_samples=5000,
                                 model_regressor=Ridge(alpha=1.0, fit_intercept=True, random_state=None))

# retrieve values
feat_names = []
feat_imports = []
for i, value in enumerate(exp.local_exp[0]):
    feat_names.append(exp.domain_mapper.feature_names[value[0]] + "-")
    feat_imports.append(abs(value[1]))

for t in range(0, 50):

    plt.figure(figsize=(5, 10))
    plt.rc('font', size=14)
    plt.rc('axes', titlesize=16)
    # plt.rc('xtick', labelsize=10)
    # plt.rc('ytick', labelsize=10)


    feat_names_ts = []
    feat_imports_ts = []

    for i, value in enumerate(feat_names):
        if f't-{t}-' in value:
            feat_names_ts.append(feat_names[i])
            feat_imports_ts.append(float(feat_imports[i]))

    # sorting
    sorted_index = np.argsort(np.array(feat_imports_ts))
    feat_imports_sorted = np.array(feat_imports_ts)[sorted_index]
    feat_names_sorted = np.array(feat_names_ts)[sorted_index]
    feat_names_sorted = feat_names_sorted.tolist()
    feat_imports_sorted = feat_imports_sorted.tolist()

    # filter
    feat_names_sorted = feat_names_sorted[19:]  # total 39 features
    feat_imports_sorted = feat_imports_sorted[19:]
    y_pos = np.arange(len(feat_names_sorted))

    # clean names
    feat_names_sorted = [n.replace(f'_t-{t}-', "") for n in feat_names_sorted]

    # colouring depending on bar length
    cmap = plt.get_cmap('Reds')
    norm = plt.Normalize(min(feat_imports_sorted), max(feat_imports_sorted))
    colors = cmap(norm(feat_imports_sorted))

    data = pd.DataFrame({'x': feat_names_sorted, 'y': y_pos})
    plot = plt.barh(y_pos, feat_imports_sorted, color=colors, zorder=2)

    # plt.yticks(y_pos, feat_names_sorted)
    plt.tick_params(left=False, labelleft=False)
    plt.xticks(np.arange(0, 1.41, step=0.2))
    plt.grid(True, zorder=0)

    def autolabel(plot):
        for idx, rect in enumerate(plot):
            if feat_imports_sorted[idx] > 0.7:
                plt.text(0.005, idx - 0.25, feat_names_sorted[idx], color='white')
            else:
                plt.text(0.005, idx - 0.25, feat_names_sorted[idx], color='black')

    autolabel(plot)

    plt.xlabel("Importance")
    plt.ylabel("Medical indicator")
    plt.title(f"Importance of medical indicators")
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig(f'../../plots/global_feat_importance_lime_{t}.{file_format}', dpi=100, bbox_inches="tight")
    plt.close(fig1)

exp.save_to_file("output.html")
