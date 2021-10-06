import matplotlib.pyplot as plt
import matplotlib
import itertools
import seaborn as sns
import numpy as np
import pickle

data_set = "sepsis"
mode = "complete"
target_activity = "Release A"


with open(f'../../output/{data_set}_{mode}_{target_activity}_shap.npy', 'rb') as f: X_all = pickle.load(f)


matplotlib.style.use('default')
matplotlib.rcParams.update({'font.size': 16})


def my_palplot(pal, size=1, ax=None):
    n = 5
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(n * size, size))
    ax.imshow(np.arange(n)[::-1].reshape(n, 1),
              cmap=matplotlib.colors.ListedColormap(list(pal)),
              interpolation="nearest", aspect="auto")


if data_set == "sepsis":

    shap_values = [
        'SHAP Leucocytes',
        'SHAP CRP',
        'SHAP LacticAcid',
        'SHAP ER Triage',
        # 'SHAP ER Sepsis Triage',
        'SHAP IV Liquid',
        'SHAP IV Antibiotics'
        # 'SHAP Admission NC',
        # 'SHAP Admission IC',
        # 'SHAP Return ER',
        # 'SHAP Release A',
        # 'SHAP Release B',
        # 'SHAP Release C',
        # 'SHAP Release D',
        # 'SHAP Release E',
    ]

elif data_set == "mimic":

    shap_values = [
        'SHAP admission_type',
        'SHAP insurance',
        'SHAP marital_status',
        'SHAP age',
        'SHAP age_dead'
    ]

else:
    print("Data set not available!")

fig11 = plt.figure(figsize=(16, 14), constrained_layout=False)  # 16, 8
grid = fig11.add_gridspec(6, 3, width_ratios=[2, 20, 0.2], wspace=0.2, hspace=0.0)  # 3,3

for i, c in enumerate(shap_values):
    ax = fig11.add_subplot(grid[i, 1])
    ax.set_xlim([-0.1, 0.1])  # -0.1, 0.1
    col = c.replace('SHAP ', '')
    X_tmp = X_all[X_all[col] > 0.]
    bins = np.linspace(X_tmp[col].min(), X_tmp[col].max(), 5)
    digitized = np.digitize(X_tmp[col], bins)

    palette = itertools.cycle(sns.color_palette("viridis"))
    for b in np.unique(digitized):
        # if seed:
        #    X_dat = X_tmp[digitized == b].sample(frac=0.2, replace=False, random_state=seed_val)
        # else:
        X_dat = X_tmp[digitized == b].sample(frac=0.2, replace=False, random_state=None)
        sns.swarmplot(data=X_dat, x=c, color=next(palette), alpha=1., size=4, ax=ax)
    [s.set_visible(False) for s in ax.spines.values()]
    if i != (len(shap_values) - 1):
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    else:
        # ax.arrow(0., 0., 1., 0.)
        # ax.arrow(0., 0., -2., 0.)
        ax.set_xlabel('SHAP Value (Effect on Model Output)')
        # ax.set_xticklabels(
        # ['-1\n(Euglycemia)', '-0.5', '-0.25', '0', '2', '4', '6', '8\n(Hypoglycemia)'])
    ax = fig11.add_subplot(grid[i, 0])
    ax.text(0, 0.3, col)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    [s.set_visible(False) for s in ax.spines.values()]
    # if i == (len(top_n_shap_values) - 1):
    #     ax.text(0.0, -1.0, 'Likely EU')

ax = fig11.add_subplot(grid[1:-1, 2])
my_palplot(sns.color_palette("viridis"), ax=ax)
ax.text(-4.2, 5.6, '   Low\nFeature\n  Value')  # 6.9
ax.text(-4.2, -1.2, '  High\nFeature\n  Value')  # -1.2
# ax.text(0.0, 5.5, 'Likely Hypo')
ax.set_yticks([])
ax.set_xticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])

fig11.tight_layout()
plt.savefig(f'../../plots/{target_activity}_shap.svg', bbox_inches="tight")