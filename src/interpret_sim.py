from src.data import get_sim_data
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

model = torch.load(os.path.join("../model", f"model_sim"))
x_seqs, x_statics, y, _, seq_features, static_features = get_sim_data('Label', 'Simulation_data_1000.csv')
case = -1

# Create dataset without prefixes
x_seqs_final = np.zeros((len(x_seqs), 12, len(x_seqs[0][0])))
x_statics_final = np.zeros((len(x_seqs), len(x_statics[0])))
for i, x in enumerate(x_seqs):
    x_seqs_final[i, :len(x), :] = np.array(x)
    x_statics_final[i, :] = np.array(x_statics[i])

"""
# (1) Print static features (global, history)
stat_numeric = ["Age", "BMI"]
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
            g = sns.lineplot(data=data, y="y", x="x", linewidth=2, color="black", ax=axes[0])
            g.axhline(y=0, color="grey", linestyle="--")

        else:
            # categorical
            g = sns.lineplot(data=data_agg, y="y_unique", x="x_unique", linewidth=2, drawstyle="steps-mid", color="black", ax=axes[0])
            g.axhline(y=0, color="grey", linestyle="--")
            axes[0].set_xticks([0, 1])

        # set ylim
        axes[0].set_ylim(-0.02, 0.22)

        g1 = sns.barplot(data=data_agg, y="x_count", x="x_unique", color="grey", ax=axes[1])
        g1.tick_params(axis="x", bottom=False, labelbottom=False)

        x_case = x_statics_final[case, idx]
        y_case = torch.from_numpy(x_case.reshape(-1, 1)).float()
        _, out_ = model.plot_feat_stat_effect(idx, y_case)
        out_ = out_.detach().numpy()[0]

        g.plot(x_case, out_ + out_delta, marker='x', ms=10, mew=2, ls='none', color='red')

        axes[0].set_title(f"Static medical indicator: {static_features[idx]}")
        axes[0].set_xlabel(None)
        axes[0].grid(True)
        axes[0].set_ylabel("Effect on prediction")

        axes[1].set_ylabel(None)
        axes[1].set_xlabel(f"{static_features[idx]}")

        f.tight_layout(pad=1.0)
        plt.savefig(f'../plots/simulation/stat_feat_{value}.png', dpi=100, bbox_inches="tight")
        plt.show()
        plt.close(f)


# (2) Print sequential features (local, history, manipulated sequence)
for t in range(1, 13):
    for idx, value in enumerate(seq_features):
        if value == "Heart Rate" or value == "Blood Pressure":

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

            g = sns.lineplot(data=data, y="y", x="x", linewidth=2, color="black")
            g.axhline(y=0, color="grey", linestyle="--")

            # set ylim
            plt.ylim(-0.02, 0.35)

            # add x value of step n
            x_step_n = x_seqs_final[case, t - 1, idx]
            y_step_n = torch.from_numpy(x_seqs_final[case, 0:t, idx].reshape(-1, t, 1)).float()
            _, out_, _, _ = model.plot_feat_seq_effect(idx, y_step_n, history=True)
            out_ = out_.detach().numpy()[0]

            g.plot(x_step_n, out_ + out_delta, marker='x', ms=10, mew=2, ls='none', color='red')

            plt.grid(True)

            plt.xlabel(f"{str(seq_features[idx])} value")
            plt.ylabel("Effect on prediction")
            plt.title("Sequential medical indicator: %s" % (str(seq_features[idx])))
            f = plt.gcf()
            plt.show()
            plt.draw()
            f.savefig(f'../plots/simulation/seq_feat_{value}_{t}.png', dpi=100)
            plt.close(f)

# prediction at step n
# print(torch.sigmoid(model(torch.from_numpy(x_seqs_final[case, :, :].reshape(1, 50, 16)),
#                          torch.from_numpy(x_statics_final[case, :].reshape(1, 23)))))


# (3) Print sequential feature transition (local, history, manipulated sequence)
for t in range(3, 13):
    for idx, value in enumerate(seq_features):

        if value == "Heart Rate" or value == "Blood Pressure":

            plt.plot(figsize=(7.5, 5))
            plt.rc('font', size=14)
            plt.rc('axes', titlesize=16)


            # steps 1 to n-2
            x_n_min_2_ = torch.from_numpy(x_seqs_final[case, 0:t - 2, idx].reshape(1, t - 2, 1)).float()
            x_n_min_2_ = x_n_min_2_.repeat(200, 1, 1)

            # steps n-1 and n
            x_n_min_1_ = torch.linspace(0, 1, 200).reshape(200, 1, 1).float()
            x_n_ = torch.linspace(0, 1, 200).reshape(200, 1, 1).float()

            # cat step n-1 and steps 1 to n-2
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

            cmap = plt.cm.get_cmap('RdBu')
            cmap = cmap.reversed()
            plt.imshow(diffs, cmap=cmap, interpolation='nearest', origin='lower', vmin=-0.4, vmax=0.4)
            cbar = plt.colorbar(label="Change of effect on prediction")

            ticks = np.linspace(0, 1, 5)
            tick_indices = np.linspace(0, len(x_n) - 1, len(ticks)).astype(int)

            plt.xticks(ticks=tick_indices, labels=ticks)
            plt.yticks(ticks=tick_indices, labels=ticks)

            # add current patient
            x_n = np.ravel(x_seqs_final[case, t-1, idx].reshape(1, 1, 1))[0]
            x_n_min_1 = np.ravel(x_seqs_final[case, t-2, idx].reshape(1, 1, 1))[0]
            plt.plot(x_n_min_1*(200-1), x_n*(200-1), marker='x', ms=10, mew=2, ls='none', color='black')

            plt.xlabel(f"{value} value (previous)", )
            plt.ylabel(f"{value} value (current)")
            plt.title(f"Sequential medical indicator: {seq_features[idx]}")
            f = plt.gcf()
            plt.show()
            plt.draw()
            f.savefig(f'../plots/simulation/seq_feat_diffs_{value}_{t}.png', dpi=100, bbox_inches="tight")
            plt.close(f)
"""

# (5) Print sequential feature (local, history)
max_len = 12
last_step = 4
seq_features_rel = ['Blood Pressure', 'Heart Rate']

for idx, value in enumerate(seq_features):
    if value in seq_features_rel:

        list_effect, list_value, list_time = [], [], []
        effect_feature_values, data_feature_values = [], []

        plt.rcParams["figure.figsize"] = (12, 5)
        plt.rc('font', size=14)
        plt.rc('axes', titlesize=16)

        for t in range(1, max_len+1):

            x, out, h_t, out_coef = model.plot_feat_seq_effect(
                idx, torch.from_numpy(x_seqs_final[case, 0:t, idx].reshape(1, t, 1)).float(), history=True)
            x = x.detach().numpy().squeeze()
            out = out.detach().numpy()

            # correction
            if t == 1:
                # step n
                x_n = torch.linspace(0, 1, 200).reshape(200, 1, 1).float()

            else:
                # step 1 to n-1
                x_hist = torch.from_numpy(x_seqs_final[case, 0:t - 1, idx].reshape(1, t - 1, 1)).float()
                x_hist = x_hist.repeat(200, 1, 1)

                # step n
                x_n = torch.linspace(0, 1, 200).reshape(200, 1, 1).float()

                # cat step n and steps 1 to n-1
                x_n = torch.cat((x_hist, x_n), 1)

            x_n, out_n, _, _ = model.plot_feat_seq_effect(idx, x_n, history=True)
            x_n = x_n.detach().numpy().squeeze()
            out_n = out_n.detach().numpy()

            out = np.squeeze(out)
            out_n = np.squeeze(out_n)
            out_n_min = min(out_n)
            out_n_delta = 0 - out_n_min
            out = out + out_n_delta

            effect_feature_values.append(out)

            if t == 1:
                data_feature_values.append(x.item())
            else:
                data_feature_values.append(x[-1])

        list_effect = list_effect + effect_feature_values
        list_value = list_value + data_feature_values
        list_time = list_time + list(range(1, max_len+1))

        data = pd.DataFrame({'x': list_time, 'y': list_value, 'z': list_effect})

        g = sns.lineplot(data=data[0:last_step], y="y", x="x", linewidth=2, color="black", zorder=1, linestyle="--")
        g.axhline(y=0, color="grey", linestyle="--")

        plt.ylim(-0.01, 0.14)

        cmap = plt.cm.get_cmap('RdBu')
        cmap = cmap.reversed()

        sc = plt.scatter(list(range(1, max_len + 1))[0:last_step], data_feature_values[0:last_step], c=effect_feature_values[0:4], cmap=cmap,
                         zorder=2, vmin=-2.5, vmax=2.5, edgecolors='black', s=100)
        plt.colorbar(sc, label='Effect on prediction')

        plt.grid(True)
        plt.ylabel(f"{str(seq_features[idx])} value")
        plt.xlabel("Measurement date")

        plt.xticks(np.arange(0, 12, step=1), ["2014-09-18 13:46", "2014-09-18 13:55", "2014-09-18 13:56", "2014-09-18 14:11",
                                              "2014-09-18 14:12", "2014-09-18 14:14", "2014-09-18 15:44", "2014-09-18 17:57",
                                              "2014-09-18 17:59", "2014-09-18 18:15", "2014-09-18 18:17",
                                              "2014-09-18 18:18"][0:last_step]+["..."]*(max_len-last_step), rotation=20)
        plt.title(f"Recent development of {str(seq_features[idx])} values")

        f = plt.gcf()
        f.tight_layout()
        plt.show()
        f.savefig(f'../plots/simulation/seq_feat_case_{case}_{value}_time.png', dpi=100, bbox_inches="tight")

        plt.close(f)