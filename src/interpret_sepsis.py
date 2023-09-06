import torch
import os
import matplotlib.pyplot as plt
import src.data as data
from src.main import time_step_blow_up
import numpy as np
import pandas as pd
import seaborn as sns
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

name = "4_98"
model = torch.load(os.path.join("../model", f"model_{name}"), map_location=torch.device('cpu'))
interactions_seq = model.get_interactions_seq()
number_interactions_seq = len(interactions_seq)

x_seqs, x_statics, y, x_time_vals_final, seq_features, static_features = data.get_sepsis_data('Admission IC', 50, 3)
x_seqs_final, x_statics_final, y_final = time_step_blow_up(x_seqs, x_statics, y, 50)


"""
# (1) Print static features (global, history)
stat_numeric = ["Age"]
for idx, value in enumerate(static_features):

	x, out = model.plot_feat_stat_effect(idx, torch.from_numpy(x_statics_final[:, idx].reshape(-1, 1)).float())
	x = x.detach().numpy().squeeze()
	out = out.detach().numpy()

	f, axes = plt.subplots(2, 1, figsize=(7.5, 5), gridspec_kw={'height_ratios': [4, 1]})
    plt.rc('font', size=14)
    plt.rc('axes', titlesize=16)

	# correction
	out = np.squeeze(out)
	out_min = min(out)
	out_delta = 0 - out_min
	out = [x + out_delta for x in out]

	data = pd.DataFrame({'x': x, 'y': out})

	if len(data["y"].unique()) > 1:

		data_agg = pd.DataFrame({'x_unique': data["x"].unique(), 'x_count': data["x"].value_counts(), 'y_unique': data["y"].unique()})

		if value in stat_numeric:
			# numerical
			g = sns.lineplot(data=data, y="y", x="x", linewidth=2, color="steelblue", ax=axes[0])
			g.axhline(y=0, color="grey", linestyle="--")

		else:
			# categorical
			g = sns.lineplot(data=data_agg, y="y_unique", x="x_unique", linewidth=2, drawstyle="steps-mid", color="steelblue", ax=axes[0])
			g.axhline(y=0, color="grey", linestyle="--")
			axes[0].set_xticks([0, 1])

		g1 = sns.barplot(data=data_agg, y="x_count", x="x_unique", color="steelblue", ax=axes[1])
		g1.tick_params(axis="x", bottom=False, labelbottom=False)

		axes[0].set_title(f"Static feature: {static_features[idx]}")
		axes[0].set_xlabel(None)
		axes[0].grid(True)
		axes[0].set_ylabel("Feature effect on model output")

		axes[1].set_ylabel(None)
		axes[1].set_xlabel("Feature value")

		f.tight_layout(pad=1.0)
		plt.savefig(f'../plots/sepsis/stat_feat_{value}.pdf', dpi=100, bbox_inches="tight")
		plt.show()
		plt.close(f)
"""

"""
# (2) Print sequential features (local, history, manipulated sequence)
case = -1
for t in range(1, 13):
    for idx, value in enumerate(seq_features):
        if value == "Leucocytes" or value == "LacticAcid" or value == "CRP":

            plt.plot(figsize=(7.5, 5))
            plt.rc('font', size=14)
            plt.rc('axes', titlesize=16)

            if t == 1:
                # step n
                inputs_ = torch.linspace(0, 1, 200).reshape(200, 1, 1).float()
                inputs = inputs_
            else:
                # step 1 to n-1
                inputs_trace =  torch.from_numpy(x_seqs_final[case, 0:t-1, idx].reshape(1, t-1, 1)).float()
                inputs_trace = inputs_trace.repeat(200, 1, 1)

                # step n
                inputs_ = torch.linspace(0, 1, 200).reshape(200, 1, 1).float()

                # cat step n and steps 1 to n-1
                inputs = torch.cat((inputs_trace, inputs_), 1)

            x, out, h_t, out_coef = model.plot_feat_seq_effect(idx, inputs, history=True)
            x = x.detach().numpy().squeeze()
            out = out.detach().numpy()

            # correction
            out = np.squeeze(out)
            out_min = min(out)
            out_delta = 0 - out_min
            out = [x + out_delta for x in out]

            inputs_ = inputs_.reshape(200,1)

            # plot x values of step n
            if len(x.shape) > 1:
                x = x[:,-1]

            data = pd.DataFrame({'x': x, 'y': out})

            g = sns.lineplot(data=data, y="y", x="x", linewidth=2, color="steelblue")
            g.axhline(y=0, color="grey", linestyle="--")

            # add x value of step n
            x_step_n = x_seqs_final[case, t - 1, idx]
            y_step_n = torch.from_numpy(x_seqs_final[case, 0:t, idx].reshape(-1, t, 1)).float()
            _, out_, _, _ = model.plot_feat_seq_effect(idx, y_step_n, history=True)
            out_ = out_.detach().numpy()[0]

            plt.plot(x_step_n, out_+out_delta, marker='s', ms=4, mew=2, ls='none', color='black')

            plt.grid(True)

            plt.xlabel("Feature value")
            plt.ylabel("Feature effect on model output")
            plt.title("Sequential feature: %s" % (str(seq_features[idx])))
            f = plt.gcf()
            plt.show()
            plt.draw()
            f.savefig(f'../plots/sepsis/seq_feat_{value}_{t}.pdf', dpi=100)
            plt.close(f)

# prediction at step n
# print(torch.sigmoid(model(torch.from_numpy(x_seqs_final[case, :, :].reshape(1, 50, 16)),
#                          torch.from_numpy(x_statics_final[case, :].reshape(1, 23)))))
"""

"""
# (3) Print sequential feature transition (local, history, manipulated sequence)
case = -1
for t in range(3, 13):
    for idx, value in enumerate(seq_features):

        if value == "Leucocytes" or value == "LacticAcid" or value == "CRP":

            plt.plot(figsize=(7.5, 5))
            plt.rc('font', size=14)
            plt.rc('axes', titlesize=16)


            # steps 1 to n-2
            x_n_min_2_ = torch.from_numpy(x_seqs_final[case, 0:t - 2, idx].reshape(1, t - 2, 1)).float()
            x_n_min_2_ = x_n_min_2_.repeat(200, 1, 1)

            # steps n-1 and n
            x_n_min_1_ = torch.linspace(0, 1, 200).reshape(200, 1, 1).float()
            x_n_ = torch.linspace(0, 1, 200).reshape(200, 1, 1).float()

            # cat step n-1 and steps 1 to n-1
            x_n_min_1 = torch.cat((x_n_min_2_, x_n_min_1_), 1)

            # cat step n and step n-1 and steps 1 to n-2
            x_n = torch.cat((x_n_min_2_, x_n_min_1_, x_n_), 1)

            # n-1
            x_n_min_1, out_n_min_1, _, _ = model.plot_feat_seq_effect(idx, x_n_min_1, history=True)
            x_n_min_1 = x_n_min_1.detach().numpy().squeeze()
            out_n_min_1 = out_n_min_1.detach().numpy().squeeze()

            # n
            x_n, out_n, _, _ = model.plot_feat_seq_effect(idx, x_n, history=True)
            x_n = x_n.detach().numpy().squeeze()
            out_n = out_n.detach().numpy().squeeze()

            diffs = np.empty((len(out_n), len(out_n)))

            for i in range(200):
                for j in range(200):
                    output1 = out_n_min_1[i]
                    output2 = out_n[j]
                    diffs[i, j] = float(output2) - float(output1)

            plt.imshow(diffs, cmap='viridis', interpolation='nearest', origin='lower')
            cbar = plt.colorbar(label='$\Delta$ Feature effect')


            ticks = np.linspace(0, 1, 5)
            tick_indices = np.linspace(0, len(x_n) - 1, len(ticks)).astype(int)

            plt.xticks(ticks=tick_indices, labels=ticks)
            plt.yticks(ticks=tick_indices, labels=ticks)

            plt.xlabel("Feature value $(t=n)$", )
            plt.ylabel("Feature value $(t=n-1)$")
            plt.title(f"Sequential feature: {seq_features[idx]}")
            f = plt.gcf()
            plt.show()
            plt.draw()
            f.savefig(f'../plots/sepsis/seq_feat_diffs_{value}_{t}.pdf', dpi=100, bbox_inches="tight")
            plt.close(f)
"""
        
"""
# (4) Print sequential feature interactions (global, no history)
print(interactions_seq)
plt.rcParams["figure.figsize"] = (7.5, 5)
from scipy.interpolate import griddata

if number_interactions_seq > 0:
    for idx in range(0, number_interactions_seq):
        fig, ax = plt.subplots()

        a = torch.linspace(0, 1, 200).reshape(200, 1, 1).float()
        b = torch.linspace(0, 1, 200).reshape(200, 1, 1).float()
        x = torch.cat((a, b), dim=2)

        X_seq, out = model.plot_feat_seq_effect_inter(idx, x)
        X_seq = X_seq.detach().numpy().squeeze()
        out = out.detach().numpy()

        a = a.detach().numpy().squeeze()
        b = b.detach().numpy().squeeze()

        grid_x, grid_y = np.mgrid[min(a):max(a):200j, min(b):max(b):200j]
        grid_z = griddata((a, b), out, (grid_x, grid_y), method="nearest")

        plt.rc('font', size=16)
        plt.rc('axes', titlesize=18)

        im = ax.imshow(grid_z.T.squeeze(), extent=(min(a), max(a), min(b), max(b)), origin='lower', cmap='magma')
        fig.colorbar(im, ax=ax, label='Interaction effect')

        plt.xlabel(f"Feature value {seq_features[interactions_seq[idx][0]]}")
        plt.ylabel(f"Feature value {seq_features[interactions_seq[idx][1]]}")
        plt.title("Interaction: %s x %s" % (
            str(seq_features[interactions_seq[idx][0]]),
            str(seq_features[interactions_seq[idx][1]])))

        fig1 = plt.gcf()
        plt.show()
        plt.draw()
        fig1.savefig(f'../plots/sepsis/seq_feat_inter_{interactions_seq[idx][0]}-{interactions_seq[idx][1]}.pdf',
                     dpi=100, bbox_inches="tight")
        plt.close(fig1)
"""

# (5) Print sequential feature (local, history)
case = -1
max_len = 12
seq_features_rel = ['Leucocytes', 'CRP', 'LacticAcid']

for idx, value in enumerate(seq_features_rel):
    if value in seq_features_rel:

        list_effect, list_value, list_time = [], [], []
        effect_feature_values, data_feature_values = [], []

        plt.rcParams["figure.figsize"] = (16, 5)
        plt.rc('font', size=14)
        plt.rc('axes', titlesize=16)

        for t in range(1, max_len+1):

            x, out, h_t, out_coef = model.plot_feat_seq_effect(
                idx, torch.from_numpy(x_seqs_final[case, 0:t, idx].reshape(1, t, 1)).float(), history=True)
            x = x.detach().numpy().squeeze()
            out = out.detach().numpy()

            # todo: correction
            # out_correction = out[0][0]  # + correction_value

            effect_feature_values.append(out)

            if t == 1:
                data_feature_values.append(x.item())
            else:
                data_feature_values.append(x[-1])


        list_effect = list_effect + effect_feature_values
        list_value = list_value + data_feature_values
        list_time = list_time + list(range(1, max_len+1))

        data = pd.DataFrame({'x': list_time, 'y': list_value, 'z': list_effect})

        g = sns.lineplot(data=data, y="y", x="x", linewidth=2, color="steelblue")

        g.axhline(y=0, color="grey", linestyle="--")

        sc = plt.scatter(list(range(1, max_len+1)), data_feature_values, c=effect_feature_values, cmap='viridis')
        
        # todo: annotations

        plt.colorbar(sc, label='Feature effect on model output')
        plt.grid(True)
        plt.ylabel("Feature value")
        plt.xlabel("Time step")
        plt.title("Sequential feature effect over time: %s" % (str(seq_features[idx])))

        f = plt.gcf()
        plt.show()
        f.savefig(f'../plots/sepsis/seq_feat_case_{case}_{value}_time.pdf', dpi=100, bbox_inches="tight")
        plt.close(f)

"""
# (5) Print sequential feature (local, history)
plt.rcParams["figure.figsize"] = (16, 16)
plt.rc('font', size=16)
plt.rc('axes', titlesize=18)
case = -1

seq_features_rel = ['Leucocytes', 'CRP', 'LacticAcid']

list_effect = []
list_value = []
list_time = []

for idx, value in enumerate(seq_features_rel):

    effect_feature_values = []
    data_feature_values = []

    if value in seq_features_rel:
        for t in range(0, 11):
            x, out, h_t, out_coef = model.plot_feat_seq_effect(
                idx, torch.from_numpy(x_seqs_final[case, 0:t + 1, idx].reshape(1, t + 1, 1)).float(), history=True)
            x = x.detach().numpy().squeeze()
            out = out.detach().numpy()

            out_correction = out[0][0]  # + correction_value

            effect_feature_values.append(out_correction)
            if t == 0:
                data_feature_values.append(x.item())
            else:
                data_feature_values.append(x[-1])

        steps = list(range(1, 12))
        for i in range(len(steps)):

            if i == 1:
                pass
            else:
                if effect_feature_values[i - 1] == effect_feature_values[i]:
                    pass
                else:
                    pass
                    if value == 'LacticAcid':
                        y_up = 0.08 if i < 3 else -0.04
                        #plt.annotate(round(effect_feature_values[i], 3),
                        #             (steps[i] - 0.2, data_feature_values[i] + y_up), color= colors[idx])
                    elif value == "Leucocytes":
                        y_up = 0.05 if i < 4 else 0.02
                        #plt.annotate(round(effect_feature_values[i], 3), (steps[i] -0.2, data_feature_values[i] + y_up), color= colors[idx])
                    else:
                        #plt.annotate(round(effect_feature_values[i], 3), (steps[i] -0.2, data_feature_values[i] + 0.02), color= colors[idx])
                        pass

        list_effect = list_effect + effect_feature_values
        list_value = list_value + data_feature_values
        list_time = list_time + list(range(1, 12))

df = pd.DataFrame({'x': list_time, 'y': list_value, 'z': list_effect})

clim_max = df['z'].max()
clim_min = df['z'].min()

fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
idx = 0

for ax in axes.flat:
    plt.sca(ax)
    effect_feature_values = []
    data_feature_values = []

    for t in range(0, 11):
        x, out, h_t, out_coef = model.plot_feat_seq_effect(
            idx, torch.from_numpy(x_seqs_final[case, 0:t + 1, idx].reshape(1, t + 1, 1)).float(), history=True)
        x = x.detach().numpy().squeeze()
        out = out.detach().numpy()

        out_correction = out[0][0]
        effect_feature_values.append(out_correction)
        if t == 0:
            data_feature_values.append(x.item())
        else:
            data_feature_values.append(x[-1])

    feature_name = seq_features_rel[idx]

    plt.plot(list(range(1, 12)), data_feature_values, linewidth=1, color='grey')
    plt.scatter(list(range(1, 12)), data_feature_values, c=effect_feature_values, cmap='viridis')
    # plt.ylim(-0.02, 0.85)
    plt.clim(clim_min, clim_max)
    plt.ylabel("Feature value of " + feature_name, fontsize=16)

    steps = list(range(1, 12))
    for i in range(len(steps)):

        if feature_name == "Leucocytes": #Done
            y_up = 0.05 if data_feature_values[i] < 0.2 else -0.09

        elif feature_name == "CRP": #WIP
            y_up = 0.08 if data_feature_values[i] < 0.4 else -0.15

        elif feature_name == 'LacticAcid': #Done
            y_up = 0.04 if data_feature_values[i] < 0.2 else -0.07

        plt.annotate(round(effect_feature_values[i], 3), (steps[i] - 0.3, data_feature_values[i] + y_up))

    plt.axhline(y=0, color='grey', linewidth=0.6)

    if idx == 2:
        plt.xlabel("Time step", fontsize=16)
        plt.xticks(np.arange(1, 12, 1))

    if idx == 0:
        plt.title(f"Sequential feature effect over time for individual patient pathway", fontsize=16)
    idx = idx + 1

plt.colorbar(ax=axes.ravel().tolist(), label='Feature effect on model output', location='right', shrink=0.8)
plt.show()
fig.savefig(f'../plots/sepsis/seq_feat_case_{case}_single_with_hist.pdf', dpi=100, bbox_inches="tight")
plt.close(fig)
"""