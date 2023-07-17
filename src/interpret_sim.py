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

# (1) Sequential features (2 time steps, without history)

def delta(y2, y1):
    return y2 - y1


t_x = list(range(0, 11))
t_y = list(range(1, 12))

plt.rcParams["figure.figsize"] = (7.5, 5)
for idx, feature in enumerate(seq_features):

    #torch.from_numpy(x_seq_final[:, t_x[t], idx].reshape(-1, 1, 1)).shape

    inputs = torch.linspace(-1,1,100).reshape(100,1,1).float()

    x_x, out_x, _, _ = model.plot_feat_seq_effect(idx, inputs)        
    x_x = x_x.detach().numpy().squeeze()
    outputs = out_x.detach().numpy().squeeze()

    diffs = np.empty((len(inputs), len(inputs)))
    for i in range(100):
        for j in range(100):
            output1 = outputs[i]
            output2 = outputs[j]
            diffs[i, j] = float(output1) - float(output2)

    inputs = inputs.detach().numpy().squeeze()

    plt.rc('font', size=16)
    plt.rc('axes', titlesize=18)

    plt.imshow(diffs, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='$\Delta$ Feature effect')

    ticks = np.linspace(-1, 1, 5)  
    tick_indices = np.linspace(0, len(inputs) - 1, len(ticks)).astype(int) 

    plt.xticks(ticks=tick_indices, labels=ticks)
    plt.yticks(ticks=tick_indices, labels=ticks)


    #plt.xticks(ticks=[-1, 1], labels=['-1', '1'])
    #plt.yticks(ticks=[-1, 1], labels=['-1', '1'])

    if feature == 'Heart Rate': #and t == 0:
        #plt.clim(0, 0.13)
        plt.title(f"Sequential feature: {seq_features[idx]}")
    #elif feature == 'Heart Rate' and t > 0:
        #plt.clim(-0.09, 0.015)
        plt.title(f"Sequential feature: {seq_features[idx]}")
    elif feature == 'Blood Pressure':
        #plt.clim(-0.09, 0.015)
        plt.title(f"Sequential feature: {seq_features[idx]}")
    elif feature == 'Medication B':
        #plt.clim(-0.15, 0.15)
        plt.title(f"Sequential feature: Medication B")
    elif feature == 'Medication A':
        #plt.clim(-0.15, 0.15)
        plt.title(f"Sequential feature: Medication A")
    elif feature == 'ER Registration':
        #plt.clim(-0.09, 0.015)
        plt.title(f"Sequential feature: {seq_features[idx]}")
    else:
        #plt.clim(-0.5, 0.5)
        plt.title(f"Sequential feature: {seq_features[idx]}")

    #plt.xlim(-0.05, 1.05)
    #plt.ylim(-0.05, 1.05)
    #plt.plot([-0.5, 1.5], [-0.5, 1.5], color='grey', linewidth=0.6)
    plt.xlabel("Feature value $t$")
    plt.ylabel("Feature value $t+1$")
    # plt.title(f"Sequential feature: {seq_features[idx]}")
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig(f'../plots/simulation/{feature}_no_history_diffs.pdf', dpi=100)
    plt.close(fig1)

# (2) Print static features (global)
for idx, value in enumerate(static_features):

    plt.rcParams["figure.figsize"] = (7.5, 5)
    plt.rc('font', size=16)

    # x, out = model.plot_feat_stat_effect_custom(idx, 0, 1)

    dataset_inputs = torch.from_numpy(x_stat_final[:, idx].reshape(-1, 1)).float()
    #print(dataset_inputs.shape)

    #inputs = torch.linspace(dataset_inputs.min(), dataset_inputs.max(),100).reshape(100,1).float()
    #print(inputs.shape)

    x, out = model.plot_feat_stat_effect(idx, dataset_inputs)
    x = x.detach().numpy().squeeze()
    out = out.detach().numpy()
    if value == "Age":
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_out = out[sorted_indices]
        plt.plot(sorted_x, sorted_out - 0.77, linewidth=3, color='steelblue')
        plt.ylim(-0.02, 0.22)
        # plt.xlim(-0.02, 1.02)

    elif value == "BMI":
        plt.plot(x, out, color='steelblue')
        plt.ylim(-0.02, 0.22)
        # plt.xlim(-0.02, 1.02)

    elif value == "Gender":
        a, b = zip(set(x), set(np.squeeze(out)))
        x = [list(a)[0], list(b)[0]]
        out = [list(a)[1] + 0.915, list(b)[1] + 0.915]
        plt.bar(x, out, color='steelblue')
        plt.xticks(x, x)
        plt.ylim(-0.02, 0.22)

    elif value == "Foreigner":
        a, b = zip(set(x), set(np.squeeze(out)))
        x = [list(a)[0], list(b)[0]]
        out = [list(a)[1] - 0.24, list(b)[1] - 0.24]
        plt.bar(x, out, color='steelblue')
        plt.xticks(x, x)
        plt.ylim(-0.02, 0.22)
    else:
        plt.plot(x, out, color='steelblue')
    plt.xlabel("Feature value")
    plt.ylabel("Feature effect on model output")
    plt.title(f"Static feature: {static_features[idx]}")
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig(f'../plots/simulation/{value}.', dpi=100)
    plt.close(fig1)


# (3) Print sequential feature over time with value range (global)
for t in range(0, 12):
    for idx, value in enumerate(seq_features):

        plt.rcParams["figure.figsize"] = (7.5, 5)
        plt.rc('font', size=16)

        if value == "Heart Rate":
            # x, out = model.plot_feat_seq_effect_custom(idx, -2, 2)
            print(x_seq_final[:, t, idx])
            x, out, h_t, out_coef = model.plot_feat_seq_effect(idx, torch.from_numpy(
                x_seq_final[:, t, idx].reshape(-1, 1, 1)).float())
            x = x.detach().numpy().squeeze()
            out = out.detach().numpy()

            if value == "Heart Rate" or value == "Blood Pressure" or value == "ER Registration":
                sorted_indices = np.argsort(x)
                sorted_x = x[sorted_indices]
                sorted_out = out[sorted_indices]
                plt.plot(sorted_x, sorted_out, linewidth=3, color='steelblue')
            elif value == "Medication A" or value == "Medication B":
                a, b = zip(set(x), set(np.squeeze(out)))
                x = [list(a)[0], list(b)[0]]
                out = [list(a)[1], list(b)[1]]
                plt.bar(x, out, color='steelblue')
                plt.xticks(x, x)

            else:
                plt.plot(x, out, color='steelblue')

            plt.xlim(-0.02, 1.02)
            plt.ylim(-0.15, 0.17)
            plt.yticks(np.arange(-0.15, 0.17, step=0.05))
            plt.xlabel("Feature value")
            plt.ylabel("Feature effect on model output")
            plt.title("Sequential feature: %s ($t_{%s}$)" % (str(seq_features[idx]), str(t + 1)))
            fig1 = plt.gcf()
            plt.show()
            plt.draw()
            fig1.savefig(f'../plots/simulation/{value}_t{t + 1}.', dpi=100)
            plt.close(fig1)

# (4) Print sequential features (local, no history)
effect_feature_values = []
case = 8
colors = ['olivedrab', 'lightskyblue', 'steelblue', 'crimson', 'orange']
plt.gca().set_prop_cycle(color=colors)

plt.rcParams["figure.figsize"] = (6, 6)
plt.rc('font', size=13)

for idx, value in enumerate(seq_features):
    effect_feature_values.append([])
    correction_value = 0
    for t in range(0, 12):
        x, out, h_t, out_coef = model.plot_feat_seq_effect(idx, torch.from_numpy(
            x_seq_final[case, t, idx].reshape(1, 1, 1)).float())
        x = x.detach().numpy().squeeze()
        out = out.detach().numpy()


        if t == 0:
            correction_value = 0 - out[0][0]

        correction_value = 0

        out_correction = out[0][0] + correction_value

        effect_feature_values[-1].append(out_correction)

    # plt.ylim(-0.17, 0.17)

    plt.plot(list(range(1, 13)), effect_feature_values[idx], label=value, linestyle='dashed', linewidth=3, marker='o', markersize=6)
plt.axhline(y=0, color='grey', linewidth=0.6)
plt.xlabel("Time step", fontsize=16)
plt.ylabel("Feature effect on model output", fontsize=16)
plt.title(f"Feature effect over time of patient pathway {case}", fontsize=16)
fig1 = plt.gcf()
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=5, fontsize=16)
#plt.legend(loc='lower left',
#           title='Sequential feature')  # adjust based on plot
plt.xticks(np.arange(1, 13, 1))
plt.rcParams["figure.figsize"] = (9, 9)
#plt.rc('axes', titlesize=20)
#plt.rc('axes', labelsize=19)
#plt.rc('xtick', labelsize=19) 
#plt.rc('ytick', labelsize=19)
#plt.rc('legend', fontsize=19)
#plt.rc('legend', title_fontsize=19)
plt.show()
plt.draw()
plt.tight_layout()
fig1.savefig(f'../plots/simulation/seq_features_case_{case}.pdf', dpi=100)
plt.close(fig1)
