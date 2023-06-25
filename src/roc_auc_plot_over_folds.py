import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Example values for three lines, the last one without CI
cut_lengths = range(1, 13)

pwn = np.array([0.724619606040199,0.724619606040199,0.724609452704102,0.724729978242851,0.731469239967536,0.745321657229588,0.763269716149029,0.771670621328077,0.70542778776966,0.695819882020976,0.764970187578782,0.703931236073629])

pwn_min = np.array(
    [0.698541105470368, 0.698935396066044, 0.693787137075924, 0.702409567225149, 0.708129618423145, 0.717318711892824,
     0.733067080999066, 0.746529377402678, 0.678540058078624, 0.667501940978566, 0.728892622876784, 0.54433641443376])
pwn_max = np.array(
    [0.765984964256112, 0.766245591786477, 0.760626987051628, 0.77167297024412, 0.770709542424757, 0.789232126769304,
     0.812867699334445, 0.818427091275092, 0.760522570797589, 0.720993294723339, 0.809880881083952, 0.846162426148275])

pwn_no_inter = np.array([0.724883107128056,0.724883107128056,0.724857736738499,0.724820464670534,0.730155983431758,0.741732381484811,0.756149565870601,0.763399394770523,0.685181771882617,0.675039913149031,0.761613428585551,0.688455990735953])

lstm = np.array([0.712663955656858,0.712537252641939,0.711792432915457,0.713474033878989,0.722485737339986,0.746149634134819,0.76040783072898,0.760871671187046,0.695631965411124,0.687865310740752,0.74255010750381,0.684625502385415])

lr = np.array([0.697284565375051,0.697284565375051,0.697284565375051,0.697284565375051,0.703209078717134,0.714008162033013,0.729772063818897,0.74254922498024,0.682663789429629,0.663426933452988,0.708270575086621,0.664793954108374])

dt = np.array([0.685640048694571,0.685640048694571,0.685640048694571,0.685640048694571,0.688694492649767,0.694205040660563,0.707433829611717,0.704307015352981,0.62832514535067,0.636057450992372,0.716426664914298,0.657961356912544])

knn = np.array([0.63824740986324,0.63824740986324,0.63824740986324,0.63824740986324,0.640381912787217,0.641668920520964,0.64597124506568,0.647853686504283,0.618549815608177,0.626228136615566,0.698255693907828,0.615716628091924])

nb = np.array([0.693075554289266,0.693075554289266,0.693075554289266,0.693075554289266,0.70029628358544,0.714990559132222,0.740870226500043,0.752764301066521,0.678054684380357,0.642271907731068,0.723071613557797,0.651721685719904])

bar_height = np.array([147, 147, 147, 147, 147, 145, 144, 140, 129, 123, 113, 105])
bar_name = bars = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12')

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i],y[i])

def plot_line_plots(cut_lengths, means_auc, mins_auc, maxes_auc, labels):
    palet = sns.color_palette()
    matplotlib.rcParams.update({'font.size': 22})

    fig, axs = plt.subplots(2, 1, figsize=(20, 16), height_ratios=[6,2])
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
    addlabels(bar_name, bar_height)
    ax2.set_ylim(0, 165)

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
