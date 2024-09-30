from src.data import get_sim_data
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

seeds = [15] # 37, 98, 137, 245]
results = {'mse_train': list(), 'mae_train': list(), 'r2_train': list()}
max_depth = 2

for seed in seeds:
    np.random.seed(seed=seed)

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    x_seqs, x_statics, y, _, seq_features, static_features = get_sim_data('Label', 'Simulation_data_50000.csv')

    # Create dataset without prefixes
    x_seq_final = np.zeros((len(x_seqs), 12, len(x_seqs[0][0])))
    x_stat_final = np.zeros((len(x_seqs), len(x_statics[0])))
    for i, x in enumerate(x_seqs):
        x_seq_final[i, :len(x), :] = np.array(x)
        x_stat_final[i, :] = np.array(x_statics[i])

    model = DecisionTreeRegressor(max_depth=max_depth, random_state=seed)
    model.fit(x_stat_final, np.ravel(y))
    y_pred = model.predict(x_stat_final)

    results["mse_train"].append(mean_squared_error(y_true=y, y_pred=y_pred))
    results["mae_train"].append(mean_absolute_error(y_true=y, y_pred=y_pred))
    results["r2_train"].append(r2_score(y_true=y, y_pred=y_pred))

    fig = plt.figure(figsize=(40,40))
    plot_tree(model, feature_names=static_features, filled=True)
    plt.show()
    fig.savefig(f"../plots/decision_tree_sim_{max_depth}_{seed}.pdf")

    print(f'Static features: {static_features}')
    print(f'Feature importances: {model.feature_importances_}')

print(f'Train mse -- avg: {sum(results["mse_train"]) / len(results["mse_train"])} --sd: {np.std(results["mse_train"])} -- values: {results["mse_train"]}')
print(f'Train mae -- avg: {sum(results["mae_train"]) / len(results["mae_train"])} --sd: {np.std(results["mae_train"])} -- values: {results["mae_train"]}')
print(f'Train r2 -- avg: {sum(results["r2_train"]) / len(results["r2_train"])} --sd: {np.std(results["r2_train"])} -- values: {results["r2_train"]}')