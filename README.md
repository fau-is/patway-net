# Explainable process predictions in healthcare

## General
- ...
 
## Setting
- Prediction task: xx
- Data attributes: xx
- Data set: sepsis
- Metrics: xx
- Machine learning model: Bi-LSTM, RandomForest, and Decision Tree
- Encoding: one-hot encoding for LSTM; ordinal encoding for tree-based ML algorithms
- Validation: 
    - split data set into train (80%) and test (20%) 
    - split train set (80%) into sub-train set (90%) and validation set (10%) 
- HPO: yes
- HPO runs: 20
- Shuffling: no
- Seed: no

## Further details
- To ensuring reproducible results during the test stage, we set the seed flag in config.py. In detail, four seeds are set:
    - np.random.seed(1377)
    - tf.random.set_seed(1377)
    - random.seed(1377)
    - optuna.samplers.TPESampler(1377)
 - Note: reproducible results via a seed are only possible if you perform the experiments on CPU
