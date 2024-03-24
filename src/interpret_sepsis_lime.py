from lime import lime_tabular
import torch
import os
import src.data as data
from src.main import time_step_blow_up
import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Ridge, Lasso

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

name = "4_98"
baseline = "lstm"
model = torch.load(os.path.join("../../model", f"model_{baseline}_{name}"), map_location=torch.device('cpu'))

x_seqs, x_statics, y, x_time_vals_final, seq_features, static_features = data.get_sepsis_data('Admission IC', 50, 3)
x_seqs_final, x_statics_final, y_final = time_step_blow_up(x_seqs, x_statics, y, 50)
case = -1
file_format = "pdf"  # pdf, png

x_seqs_final = torch.from_numpy(x_seqs_final)
x_statics_final = torch.from_numpy(x_statics_final)
y_final = torch.from_numpy(y_final)

x_statics_final_ = torch.reshape(x_statics_final, (-1, 1, x_statics_final.shape[1]))
x_stats = copy.copy(x_statics_final_)
T = x_seqs_final.shape[1]
for t in range(1, T):
    x_stats = torch.concat((x_stats, x_statics_final_), 1)
x_seqs_final = torch.concat((x_seqs_final, x_stats), 2).float()

# x_seqs_final = np.array(x_seqs_final)

explainer = lime_tabular.RecurrentTabularExplainer(x_seqs_final, training_labels=y_final,
                                                   feature_names=seq_features + static_features,
                                                   discretize_continuous=True,
                                                   class_names=['Yes', 'No'])
                                                   # kernel_width=0.01)

exp = explainer.explain_instance(x_seqs_final[case], model, num_features=20,
                                 labels=(0,), num_samples=5000,
                                 model_regressor=Ridge(alpha=1.0, fit_intercept=True, random_state=None))

exp.save_to_file("output.html")
