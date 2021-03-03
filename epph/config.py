import os
import argparse
import epph.utils as utils


def load():
    parser = argparse.ArgumentParser()

    # General parameters
    parser.add_argument('--mode', default=0, type=int)
    """ There are three modes
        0 = train and test model
        1 = explain prediction for random test case
        2 = evaluate explanations for predictions (test set) 
    """
    # Mode 1 + 2
    parser.add_argument('--xai', default="shap", type=str)  # shap
    parser.add_argument('--shap_num_samples', default=100, type=int)  # 100 (good estimate), 1000 (very good estimate)

    # Mode 1
    parser.add_argument('--rand_lower_bound', default=5, type=int)
    parser.add_argument('--rand_upper_bound', default=5, type=int)


    # Classifier
    #   LSTM -> Bi-directional long short-term neural network
    #   RF   -> Random Forest
    #   DT   -> Decision Tree
    parser.add_argument('--classifier', default="LSTM", type=str)  # LSTM, RF or DT

    # Parameters for deep neural network
    parser.add_argument('--dnn_num_epochs', default=100, type=int)
    parser.add_argument('--eager_execution', default=True, type=utils.str2bool)  # set to avoid retracing with tf

    # Directories
    parser.add_argument('--task', default="nap")
    parser.add_argument('--data_set', default="sepsis_raw_sample_100.csv")
    parser.add_argument('--data_dir', default="./data/")
    parser.add_argument('--model_dir', default="nap/models/")
    parser.add_argument('--result_dir', default="./results/")

    # Experiments
    parser.add_argument('--run_experiments', default=True, type=utils.str2bool)
    parser.add_argument('--model_id', default=0, type=int)
    parser.add_argument('--experiments_dir', default="./experiments/")
    parser.add_argument('--experiments_file', default="experiments_template.csv")

    # Parameters for validation
    parser.add_argument('--seed', default=False, type=utils.str2bool)
    parser.add_argument('--seed_val', default=1377, type=int)
    parser.add_argument('--shuffle', default=False, type=int)
    parser.add_argument('--split_rate_train', default=0.8, type=float)
    parser.add_argument('--val_split', default=0.2, type=float)
    parser.add_argument('--batch_size_train', default=128, type=int)
    parser.add_argument('--batch_size_test', default=1, type=int)

    # hpo general
    parser.add_argument('--hpo', default=True, type=utils.str2bool)
    parser.add_argument('--hpo_eval_runs', default=5, type=int)
    parser.add_argument('--split_rate_train_hpo', default=0.9, type=float)

    # hpo LSTM
    parser.add_argument('--hpo_activation', default=['linear', 'tanh', 'relu'], type=list)
    parser.add_argument('--hpo_kernel_initializer', default=['glorot_uniform'], type=list)
    parser.add_argument('--hpo_optimizer', default=['adam', 'nadam', 'rmsprop'], type=list)

    # hpo RF + DT
    parser.add_argument('--hpo_n_estimators', default=[100, 200, 300, 500], type=list)
    parser.add_argument('--hpo_criterion', default=['gini', 'entropy'], type=list)
    parser.add_argument('--hpo_min_samples_split', default=[2, 3, 4], type=list)
    parser.add_argument('--hpo_max_depth', default=[5, 10, 20, 50], type=list)

    # Pre-processing
    parser.add_argument('--encoding_num', default="min_max_norm", type=str)
    parser.add_argument('--encoding_cat', default="onehot", type=str)  # onehot, hash or int for categorical attributes
    parser.add_argument('--num_hash_output', default=10, type=int)

    # pm4py
    parser.add_argument('--case_id_key', default="case_id", type=str)
    parser.add_argument('--activity_key', default="activity", type=str)

    # Parameters for gpu processing
    parser.add_argument('--gpu_ratio', default=0.2, type=float)
    parser.add_argument('--cpu_num', default=1, type=int)
    parser.add_argument('--gpu_device', default="0", type=str)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device  # "-1"

    return args
