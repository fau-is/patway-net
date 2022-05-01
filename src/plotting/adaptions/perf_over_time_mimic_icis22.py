# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 12:44:25 2021

@author: ov59opom
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Example values for three lines, the last one without CI
cut_lengths = range(1, 17)

###### AUC Values
mean_line_stat_AUC = np.array(
    [0.67749606918239, 0.67749606918239, 0.67749606918239, 0.67749606918239, 0.720993858179787, 0.720993858179787,
     0.636290322580645, 0.636290322580645, 0.778571428571428, 0.778571428571428, 0.718518518518518, 0.718518518518518,
     0.5, 0.5, 0.5125, 0.5125])
mean_line_seq_AUC = np.array(
    [0.665978773584905, 0.820252133872416, 0.818519766397124, 0.893345687331536, 0.689335566722501, 0.892015633724176,
     0.7, 0.708870967741935, 0.545535714285714, 0.950892857142857, 0.95, 0.95, 0.893333333333333, 0.893333333333333,
     0.8875, 0.775])
mean_line_RF_AUC = np.array(
    [0.72172619047619, 0.83512747079964, 0.843837039532794, 0.9275227425876, 0.783263539921831, 0.958654383026242,
     0.843817204301075, 0.844892473118279, 0.740178571428571, 0.96875, 0.95, 0.944444444444444, 0.86, 0.813333333333333,
     0.7875, 0.7])
mean_line_NB_AUC = np.array(
    [0.708936994609164, 0.808555143755615, 0.810408243486073, 0.852818957771788, 0.724734785036292, 0.752652149637074,
     0.744623655913978, 0.747311827956989, 0.580357142857142, 0.625, 0.574074074074074, 0.648148148148148,
     0.133333333333333, 0.133333333333333, 0, 0])
mean_line_LR_AUC = np.array(
    [0.723902178796046, 0.828686545372866, 0.835986635220125, 0.907176549865229, 0.743718592964824, 0.91680625348967,
     0.71505376344086, 0.75, 0.571428571428571, 0.866071428571428, 0.833333333333333, 0.833333333333333,
     0.533333333333333, 0.533333333333333, 0.5, 0.5])
mean_line_KNN_AUC = np.array(
    [0.558625336927223, 0.735371743036837, 0.737168688230008, 0.795527291105121, 0.691652707984366, 0.80178671133445,
     0.974462365591397, 0.978494623655914, 0.959821428571428, 0.950892857142857, 0.916666666666666, 0.916666666666666,
     0.8, 0.766666666666666, 0.8125, 0.8125])
mean_line_GB_AUC = np.array(
    [0.729826482479784, 0.811320754716981, 0.833195754716981, 0.899665880503144, 0.714405360134003, 0.901730876605248,
     0.682795698924731, 0.71774193548387, 0.535714285714285, 0.955357142857142, 0.944444444444444, 0.925925925925925,
     0.786666666666667, 0.786666666666667, 0.875, 0.875])
mean_line_ADA_AUC = np.array(
    [0.73378537735849, 0.826875561545372, 0.835930480682839, 0.905211141060197, 0.760329424902289, 0.903405918481295,
     0.712365591397849, 0.744623655913978, 0.571428571428571, 0.803571428571428, 0.777777777777777, 0.777777777777777,
     0.533333333333333, 0.533333333333333, 0.5, 0.5])
mean_line_complete_AUC = np.array(
    [0.71975516621743, 0.850642969451931, 0.858709568733153, 0.915491632973944, 0.749734785036292, 0.927414852037967,
     0.751075268817204, 0.776612903225806, 0.6375, 1, 1, 1, 1, 1, 1, 0.9875])

max_line_complete_AUC = np.array(
    [0.727785589502207, 0.864576923286653, 0.873276081953302, 0.924608097814432, 0.777440744089, 0.942963094851956,
     0.792636877613092, 0.824451037682913, 0.703014058313728, 1, 1, 1, 1, 1, 1, 1.025])
min_line_complete_AUC = np.array(
    [0.711724742932653, 0.83670901561721, 0.844143055513004, 0.906375168133455, 0.722028825983585, 0.911866609223978,
     0.709513660021316, 0.728774768768699, 0.571985941686271, 1, 1, 1, 1, 1, 1, 0.95])


######

def plot_line_plots(cut_lengths, means_auc, mins_auc, maxes_auc, labels):
    palet = sns.color_palette("tab10")
    matplotlib.rcParams.update({'font.size': 24})

    fig, axs = plt.subplots(figsize=(22, 10))
    ax2 = axs

    def plot_on_axes(ax, m, a, b, title):
        for i, (mean, a, b, l) in enumerate(zip(m, a, b, labels)):
            if i == 0:
                ax.plot(cut_lengths, mean, color=palet[i], label=l)
            else:
                ax.plot(cut_lengths, mean, color=palet[i], linestyle='dashed', label=l)
            if len(a) > 0 and len(b) > 0:
                ax.fill_between(cut_lengths, a, b, alpha=.2, color=palet[i])
        # ax.set_title(r'$M_{%s}$', fontsize=30)
        ax.set_xlabel('Size of Process Instance Prefix for Prediction')
        ax.set_xticks(np.arange(1, 17, step=1))
        ax.set_xlim(1, 16)
        ax.set_ylabel("AUC$_{ROC}$")
        ax.set_ylim(0.45, 1.01)
        # ax.title.set_text(title)

    plot_on_axes(ax2, means_auc, mins_auc, maxes_auc, title='AUC$_{ROC}$')

    ax2.legend(ncol=1, loc='lower left',
               columnspacing=1.3, labelspacing=0.0,
               handletextpad=0.0, handlelength=1.5,
               fancybox=False, shadow=False)

    y_axis = ax2.axes.get_yaxis()
    y_axis.set_visible(True)

    # fig.text(0.5, -0.01, 'Size of patient pathway prefix for prediction', ha='center')

    plt.tight_layout()
    plt.savefig('tmp.pdf', bbox_inches="tight")


# There are three args for acc and three args for auc
# Each arg is a list where the length equals the number of lines
# Each line is a list (or a 1D numpy array) with the y-values
# Pass an empty list in mins/maxes arg for a line not to have confidence intervals
plot_line_plots(cut_lengths,
                means_auc=[mean_line_complete_AUC, mean_line_ADA_AUC, mean_line_GB_AUC, mean_line_RF_AUC,
                           mean_line_LR_AUC],
                mins_auc=[min_line_complete_AUC, [], [], [], []],
                maxes_auc=[max_line_complete_AUC, [], [], [], []],
                labels=['HIXPred $\pm$ SD', 'AdaBoost', 'Gradient Boosting', 'Random Forest', 'Logistic Regression'])