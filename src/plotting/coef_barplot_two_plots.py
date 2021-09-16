import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Example values for coefs of two tasks
task_1 = "Release Type A (REA)"
task_2 = "Admission IC (AIC)"

coef_values_task_1 = [-0.2888972, -0.001997236, -0.23954397, -0.2559873, -0.45435518, 0.07244425, 0.03626472, -0.16619293, -0.038904637, 0.037489716, 0.1976995, 0.07379714, 0.11746674, 0.3859931, 0.2782478, 0.4835849, -0.3130332, -0.43463305, -0.294928, -0.062663764, 0.9346965, -0.12880808, 0.28425518, 0.4040266]
coef_values_task_2 = [-0.2888972, -0.001997236, -0.23954397, -0.2559873, -0.45435518, 0.07244425, 0.03626472, -0.16619293, -0.038904637, 0.037489716, 0.1976995, 0.07379714, 0.11746674, 0.3859931, 0.2782478, 0.4835849, -0.3130332, -0.43463305, -0.294928, -0.062663764, 0.9346965, -0.12880808, 0.28425518, 0.4040266]

static_features = ['InfectionSuspected', 'DiagnosticBlood', 'DisfuncOrg',
                       'SIRSCritTachypnea', 'Hypotensie',
                       'SIRSCritHeartRate', 'Infusion', 'DiagnosticArtAstrup', 'Age',
                       'DiagnosticIC', 'DiagnosticSputum', 'DiagnosticLiquor',
                       'DiagnosticOther', 'SIRSCriteria2OrMore', 'DiagnosticXthorax',
                       'SIRSCritTemperature', 'DiagnosticUrinaryCulture', 'SIRSCritLeucos',
                       'Oligurie', 'DiagnosticLacticAcid', 'Diagnose', 'Hypoxie',
                       'DiagnosticUrinarySediment', 'DiagnosticECG']

coefs_task_1 = dict(zip([x for x in static_features], np.array(coef_values_task_1)))
coefs_task_2 = dict(zip([x for x in static_features], np.array(coef_values_task_2)))


def colors_from_values(values, palette_name):
    normalized = (values - min(values)) / (max(values) - min(values))
    indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
    palette = sns.color_palette(palette_name, len(values))
    return np.array(palette).take(indices, axis=0)


def my_palplot(pal, size=1, ax=None):
    n = 5
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(n * size, size))
    ax.imshow(np.arange(n)[::-1].reshape(n, 1),
              cmap=matplotlib.colors.ListedColormap(list(pal)),
              interpolation="nearest", aspect="auto")


def plot_box_plots(coefs_1, coefs_2):
    matplotlib.rcParams.update({'font.size': 20})
    
    max_v = max(max(list(coefs_1.values())), max(list(coefs_2.values())))
    min_v = min(min(list(coefs_1.values())), min(list(coefs_2.values())))
    
    fig = plt.figure(figsize=(30, 14), constrained_layout=False)
    grid = fig.add_gridspec(1, 3, width_ratios=[10, 10, 0.2], wspace=0.2, hspace=0.0)
    
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[0, 2])
    
    def plot_on_ax(ax, coefs, title):
        coefs_values = list(coefs.values())
        coefs_names = list(coefs.keys())
        sns.barplot(x=coefs_values, y=coefs_names, orient='h', 
                    palette=colors_from_values(coefs_values, "viridis"), ax=ax)
        ax.set_xlim(min_v - 0.1*max_v, max_v + 0.1*max_v)
        ax.title.set_text(title)
    
    plot_on_ax(ax1, coefs_1, title=task_1)
    plot_on_ax(ax2, coefs_2, title=task_2)
    
    fig.text(0.5, 0.03, 'Value of corresponding coefficient', ha='center')

    my_palplot(sns.color_palette("viridis"), ax=ax3)
    ax3.text(-5.2, 5.0, 'Strong negative\nimpact on model\noutput')
    ax3.text(-5.2, -0.6, 'Strong positive\nimpact on model\noutput')
    ax3.set_yticks([])
    ax3.set_xticks([])
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])

    plt.tight_layout()
    plt.savefig('tmp.png', bbox_inches="tight")


plot_box_plots(coefs_task_1, coefs_task_2)