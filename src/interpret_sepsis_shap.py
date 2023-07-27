import torch
import os
import matplotlib.pyplot as plt
import src.data as data
from src.main import time_step_blow_up
import numpy as np
import pandas as pd
import shap

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

name = "4_98"
baseline = "xgb"
model = torch.load(os.path.join("../model", f"model_{baseline}_{name}"), map_location=torch.device('cpu'))

x_seqs, x_statics, y, x_time_vals_final, seq_features, static_features = data.get_sepsis_data('Admission IC', 50, 3)
# x_seqs_final, x_statics_final, y_final = time_step_blow_up(x_seqs, x_statics, y, 50)

x_statics = pd.DataFrame(np.array(x_statics), columns=static_features)

explainer = shap.Explainer(model, x_statics)
shap_values = explainer(x_statics)

feat = "Age"

# shap.plots.scatter(shap_values[:,"Age"])

plt.rcParams["figure.figsize"] = (7.5, 5)
plt.rc('font', size=16)
plt.scatter(x=shap_values.data[:, static_features.index(feat)], y=shap_values.values[:, static_features.index(feat)])
plt.xlabel("Feature value")
plt.ylabel("Feature effect on model output")
plt.title(f"Static feature: {feat}")
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig(f'../plots/sepsis/stat_feat_{feat}_shap.pdf', dpi=100, bbox_inches="tight")
plt.close(fig1)
