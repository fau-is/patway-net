import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Example values for three lines, the last one without CI
cut_lengths = range(1, 13)

pwn = np.array(
    [0.725431477932034, 0.725795313890441, 0.720914369797219, 0.730054396755578, 0.730340516462543, 0.747047802217598,
     0.765784786242079, 0.773455542913255, 0.707102344342321, 0.695916495640541, 0.76753827048675, 0.706033526208796])
pwn_min = np.array(
    [0.698541105470368, 0.698935396066044, 0.693787137075924, 0.702409567225149, 0.708129618423145, 0.717318711892824,
     0.733067080999066, 0.746529377402678, 0.678540058078624, 0.667501940978566, 0.728892622876784, 0.54433641443376])
pwn_max = np.array(
    [0.765984964256112, 0.766245591786477, 0.760626987051628, 0.77167297024412, 0.770709542424757, 0.789232126769304,
     0.812867699334445, 0.818427091275092, 0.760522570797589, 0.720993294723339, 0.809880881083952, 0.846162426148275])

pwn_no_inter = np.array(
    [0.725863240779113, 0.726175917988022, 0.721602738906766, 0.729847099034346, 0.72871577120561, 0.74435273202734,
     0.758927281048665, 0.763140752238913, 0.683283619507622, 0.673411510362308, 0.761565720830369, 0.696656099768952])

lstm = np.array(
    [0.71130663334024, 0.711395658277079, 0.70597047058531, 0.718229500320175, 0.71977891004647, 0.746990052316953,
     0.760438425083142, 0.759167761106017, 0.698782923716998, 0.69028173786019, 0.744411043632586, 0.695952590642131])
lr = np.array(
    [0.699334943535018, 0.699847658202834, 0.694732543279622, 0.703357767453601, 0.706186249596681, 0.71609373552486,
     0.732720483705402, 0.745733190927193, 0.687082425840912, 0.66563943874156, 0.71241855459813, 0.672832500368751])
dt = np.array(
    [0.679403310194778, 0.679458386807996, 0.675717218940479, 0.683381636441479, 0.678923988005606, 0.691488785659756,
     0.704582685427964, 0.697286051907026, 0.626819625208496, 0.63201505881774, 0.71631352362172, 0.659696567180092])
knn = np.array(
    [0.64304469280978, 0.643199084622192, 0.641544987269577, 0.644642624792476, 0.64416129720295, 0.647212000457663,
     0.651987245450017, 0.652475970966347, 0.626477067339333, 0.631557589055525, 0.700172316132425, 0.626886793479696])
nb = np.array(
    [0.69332222596353, 0.693983270992819, 0.687779578130691, 0.698606758214473, 0.695896097475087, 0.71841924333024,
     0.744777375017191, 0.753312764836698, 0.678852151722145, 0.640860289421346, 0.726566187663933, 0.652897593343812])

bar_height = np.array([5, 10, 3, 4, 1, 1, 1, 1, 1, 1, 1, 1])
bar_name = bars = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12')


def plot_line_plots(cut_lengths, means_auc, mins_auc, maxes_auc, labels):
    palet = sns.color_palette()
    matplotlib.rcParams.update({'font.size': 20})

    fig, axs = plt.subplots(2, 1, figsize=(16, 16), height_ratios=[2,1])
    ax1 = axs[0]
    ax2 = axs[1]

    def plot_on_axes(ax, m, a, b, title):
        for i, (mean, a, b, l) in enumerate(zip(means_auc, mins_auc, maxes_auc, labels)):
            ax.plot(cut_lengths, mean, color=palet[i], label=l, linewidth=2.0, linestyle='--', marker='o')

        ax.set_xlabel('Size of patient pathway prefix')
        ax.set_xticks(np.arange(1, 13, step=1))
        ax.set_ylabel(r'$AUC_{ROC}$')
        ax.set_ylim(0.57, 0.81)

    plot_on_axes(ax1, means_auc, mins_auc, maxes_auc, title='Prediction performance over time')

    ax1.legend(ncol=2, loc='lower left',
               columnspacing=1.3, labelspacing=0.0,
               handletextpad=0.0, handlelength=1.5,
               fancybox=False, shadow=False)

    ax2.bar(bar_name, bar_height, width = 0.3, color="grey")
    ax2.set_xlabel('Size of patient pathway prefix')
    ax2.set_xticks(np.arange(len(bar_name)))
    ax2.set_ylabel(f'Number of\n patient pathway prefixes')

    plt.tight_layout()
    plt.savefig(r"..\plots\prediction" + "\\" + "pred_perf_over_time.pdf")
    plt.show()


plot_line_plots(cut_lengths,
                means_auc=[pwn, pwn_no_inter, lstm, lr, dt, knn, nb],
                mins_auc=[pwn_min, [], [], [], [], [], []],
                maxes_auc=[pwn_max, [], [], [], [], [], []],
                labels=['PatWay-Net (with interaction)',
                        'PatWay-Net (without interaction)',
                        'LSTM network (with static module)',
                        'Logistic regression',
                        'Decision tree',
                        r'$K$-nearest neighbor',
                        r'Naïve Bayes'])
