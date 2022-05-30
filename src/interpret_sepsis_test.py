import torch
import os
import matplotlib.pyplot as plt
import src.data as data
from src.main import time_step_blow_up
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

repetition = 0
# map location param is required as the model was trained on gpu
model = torch.load(os.path.join("../model", f"model_{repetition}"), map_location=torch.device('cpu'))
interactions_seq = model.get_number_interactions_seq()
number_interactions_seq = len(interactions_seq)

# Note: static features include binary features created from the diagnosis feature or not
x_seqs, x_statics, y, x_time_vals_final, seq_features, static_features = data.get_sepsis_data('Admission IC', 50, 3)
x_seqs_final, x_statics_final, y_final = time_step_blow_up(x_seqs, x_statics, y, 50)


f_imp = model.plot_feat_importance()
f_imp = f_imp.tolist()
f_imp = [x[0] for x in f_imp]

res = dict()

# 164 ->  28 + (16+1)*8

# seq features
for i in range(0, len(seq_features)):
    res[seq_features[i]] = sum(f_imp[8*i:8*1+1])

# interaction features
for i in range(0, 1):
    res[seq_features[i]] = sum(f_imp[(128)+(8*i):(128)+(8*i+1)])

# static featues
for i in range(0, len(static_features)):
    res[seq_features[i]] = f_imp[136-1+i]

print(0)