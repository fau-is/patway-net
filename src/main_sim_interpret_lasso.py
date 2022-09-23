from src.data import get_sim_data
import numpy as np
import os
from sklearn.linear_model import Lasso

seed = 47
np.random.seed(seed=seed)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
x_seqs, x_statics, y, _, seq_features, static_features = get_sim_data('Label', 'Simulation_data_1000.csv')

x_seq_final = np.zeros((len(x_seqs), 12, len(x_seqs[0][0])))
x_stat_final = np.zeros((len(x_seqs), len(x_statics[0])))
for i, x in enumerate(x_seqs):
    x_seq_final[i, :len(x), :] = np.array(x)
    x_stat_final[i, :] = np.array(x_statics[i])

# https://www.kirenz.com/post/2019-08-12-python-lasso-regression-auto/
model = Lasso(alpha=0.01)
model.fit(x_stat_final, np.ravel(y))

print(static_features)
print(model.coef_)
print(model.intercept_)
