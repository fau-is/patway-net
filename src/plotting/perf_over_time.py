import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Example values for three lines, the last one without CI
cut_lengths = range(1, 21)
mean_line = np.array([0.7527173913043479, 0.7625, 0.7695652173913043, 0.7793478260869564, 0.7831521739130434, 0.7907608695652173, 0.7891304347826087, 0.8108695652173912, 0.8038043478260871, 0.8038043478260871, 0.8038043478260871, 0.8038043478260871, 0.8005434782608696, 0.8005434782608696, 0.8005434782608696, 0.7961956521739131, 0.7961956521739131, 0.7961956521739131, 0.7961956521739131, 0.7961956521739131])
min_line = np.array([0.7459239130434783, 0.7554347826086957, 0.7663043478260869, 0.7771739130434783, 0.782608695652174, 0.7880434782608695, 0.7880434782608695, 0.8097826086956522, 0.8002717391304348, 0.8002717391304348, 0.8002717391304348, 0.8002717391304348, 0.7989130434782609, 0.7989130434782609, 0.7989130434782609, 0.7934782608695652, 0.7934782608695652, 0.7934782608695652, 0.7934782608695652, 0.7934782608695652])
max_line = np.array([0.7595108695652174, 0.7717391304347826, 0.7771739130434783, 0.7880434782608695, 0.7880434782608695, 0.7934782608695652, 0.7934782608695652, 0.8152173913043478, 0.8043478260869565, 0.8043478260869565, 0.8043478260869565, 0.8043478260869565, 0.8029891304347826, 0.8043478260869565, 0.8043478260869565, 0.7989130434782609, 0.7989130434782609, 0.7989130434782609, 0.7989130434782609, 0.7989130434782609])

mean_line_2 = mean_line - 0.3
min_line_2 = mean_line_2 - 0.1 * np.random.rand(len(mean_line))
max_line_2 = mean_line_2 + 0.1 * np.random.rand(len(mean_line))

mean_line_3 = mean_line - 0.5
######


def plot_line_plots(cut_lengths, means_acc, mins_acc, maxes_acc, 
                    means_auc, mins_auc, maxes_auc, labels):
    palet = sns.color_palette("Set2")
    matplotlib.rcParams.update({'font.size': 20})
    
    fig, axs = plt.subplots(1, 2, figsize=(22, 8))
    ax1 = axs[0]
    ax2 = axs[1]
    
    def plot_on_axes(ax, m, a, b, title):
        for i, (mean, a, b, l) in enumerate(zip(means_acc, mins_acc, maxes_acc, labels)):
            ax.plot(cut_lengths, mean, color=palet[i], label=l)
            if len(a) > 0 and len(b) > 0:
                ax.fill_between(cut_lengths, a, b, alpha=.2, color=palet[i])
        # ax.set_title(r'$M_{%s}$' % target_activity_abbreviation, fontsize=30)
        # ax.set_xlabel('Size of Process Instance Prefix for Prediction')
        ax.set_xticks(np.arange(0, 20 + 1, step=5))
        ax.set_ylabel('Prediction performance')
        ax.set_ylim(0.0, 1)
        ax.title.set_text(title)
        
    plot_on_axes(ax1, means_acc, mins_acc, maxes_acc, title='Accuracy')
    plot_on_axes(ax2, means_auc, mins_auc, maxes_auc, title='Area under receiver operating characteristic (AUROC)')
    
    ax1.legend(ncol=2, loc='lower left', 
           columnspacing=1.3, labelspacing=0.0,
           handletextpad=0.0, handlelength=1.5,
           fancybox=False, shadow=False)
    # ax2.set_yticks([])
    y_axis = ax2.axes.get_yaxis()
    y_axis.set_visible(False)

    
    fig.text(0.5, -0.01, 'Size of process instance prefix for prediction', ha='center')

    plt.tight_layout()
    plt.savefig('tmp.png', bbox_inches="tight")
    
    
# There are three args for acc and three args for auc
# Each arg is a list where the length equals the number of lines
# Each line is a list (or a 1D numpy array) with the y-values
# Pass an empty list in mins/maxes arg for a line not to have confidence intervals
plot_line_plots(cut_lengths, 
                means_acc = [mean_line, mean_line_2, mean_line_3], 
                mins_acc = [min_line, min_line_2, []], 
                maxes_acc = [max_line, max_line_2, []],
                means_auc = [mean_line, mean_line_2, mean_line_3], 
                mins_auc = [min_line, min_line_2, []], 
                maxes_auc = [max_line, max_line_2, []],
                labels=[r'label a $\pm$ SD', r'label b $\pm$ SD', 'label c'])