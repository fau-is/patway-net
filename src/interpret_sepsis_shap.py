import torch
import os
import matplotlib.pyplot as plt
import src.data as data
from src.main import time_step_blow_up
import numpy as np
import pandas as pd
import shap
import seaborn as sns

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

name = "4_98"
baseline = "xgb"
model = torch.load(os.path.join("../model", f"model_{baseline}_{name}"), map_location=torch.device('cpu'))

x_seqs, x_statics, y, x_time_vals_final, seq_features, static_features = data.get_sepsis_data('Admission IC', 50, 3)
x_seqs_final, x_statics_final, y_final = time_step_blow_up(x_seqs, x_statics, y, 50)
file_format="pdf"  # pdf, png
case = -1

x_statics = pd.DataFrame(np.array(x_statics_final), columns=static_features)
explainer = shap.Explainer(model, x_statics)
shap_values = explainer(x_statics)

stat_numeric = ["Age"]
for idx, value in enumerate(static_features):

    if value == "Age":

        x= shap_values.data[:, static_features.index(value)]
        out = shap_values.values[:, static_features.index(value)]

        plt.rc('font', size=14)
        plt.rc('axes', titlesize=16)

        f, axes = plt.subplots(2, 1, figsize=(7.5, 5), gridspec_kw={'height_ratios': [4, 1]})

        # correction
        out = np.squeeze(out)
        out_min = min(out)
        out_delta = 0 - out_min
        out = [x + out_delta for x in out]

        data = pd.DataFrame({'x': x, 'y': out})

        if len(data["y"].unique()) > 1:

            data_agg = pd.DataFrame({'x_unique': data["x"].unique(), 'x_count': data["x"].value_counts()})

            if value in stat_numeric:
                # numerical
                g = sns.scatterplot(data=data, y="y", x="x", linewidth=2, color="black", ax=axes[0])
                g.axhline(y=0, color="grey", linestyle="--")

            g1 = sns.barplot(data=data_agg, y="x_count", x="x_unique", color="grey", ax=axes[1])
            g1.tick_params(axis="x", bottom=False, labelbottom=False)

            x_case = x_statics_final[case, idx]
            out_ = shap_values.values[case, static_features.index(value)]

            g.plot(x_case, out_ + out_delta, marker='x', ms=10, mew=2, ls='none', color='red')

            axes[0].set_title(f"Static medical indicator: {static_features[idx]}")
            axes[0].set_xlabel(None)
            axes[0].grid(True)
            axes[0].set_ylabel("Effect on prediction")

            axes[1].set_ylabel(None)
            axes[1].set_xlabel(f"{static_features[idx]}")

            f.tight_layout(pad=1.0)
            plt.savefig(f'../plots/sepsis/stat_feat_{value}_shap.{file_format}', dpi=100, bbox_inches="tight")
            plt.show()
            plt.close(f)