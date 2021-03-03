import argparse
import sys
import csv
import sklearn
import arrow
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from joblib import load
import random


def load_experiments(args):
    """
    Retrieves the configuration settings for all experiments.

    Parameters
    ----------
    args : Namespace
        Settings of the configuration parameters.

    Returns
    -------
    dict :
        The configuration settings for a single experiment/execution. A key is the configuration parameter (as in
        config file) and a value is the corresponding value.

    """

    exp_df = pd.read_csv(args.experiments_dir + args.experiments_file)
    exp_dicts = exp_df.to_dict('index')
    return exp_dicts


def set_experiment_config(args, exp_config):
    """
    Sets the configuration settings for a single experiment/execution.

    Parameters
    ----------
    args : Namespace
        Settings of the configuration parameters.
    exp_config : dict
        A key is the configuration parameter (as in config file) and a value is the corresponding value.

    Returns
    -------
    args : Namespace
        Updated settings of the configuration parameters.

    """

    args.data_set = exp_config["data_set"]
    args.mode = exp_config["mode"]
    args.classifier = exp_config["classifier"]
    args.model_id = exp_config["model_id"]
    args.hpo = exp_config["hpo"]
    args.hpo_eval_runs = exp_config["hpo_eval_runs"]
    args.xai = exp_config["xai"]
    args.removed_events_relevance = exp_config["removed_events_relevance"]
    args.removed_events_num = exp_config["removed_events_num"]
    args.shuffle = exp_config["shuffle"]
    args.seed = exp_config["seed"]
    args.seed_val = exp_config["seed_val"]
    args.explain_cf_c_ratio = exp_config["explain_cf_c_ratio"]

    return args


measures = {
    "accuracy_value": 0.0,
    "precision_micro_value": 0.0,
    "precision_macro_value": 0.0,
    "precision_weighted_value": 0.0,
    "recall_micro_value": 0.0,
    "recall_macro_value": 0.0,
    "recall_weighted_value": 0.0,
    "f1_micro_value": 0.0,
    "f1_macro_value": 0.0,
    "f1_weighted_value": 0.0,
    "auc_roc_value": 0.0,
    "training_time_seconds": 0.0,  # time for complete training
    "prediction_times_seconds": [],  # time for each prediction
    "explanation_times_seconds": []  # time for each explanation
}


def ll_print(message):
    """
    Prints message.
    :param message:
    :return:
    """
    sys.stdout.write(message)
    sys.stdout.flush()


def avg(numbers):
    """
    Calculates average.
    :param numbers:
    :return:
    """
    if len(numbers) == 0:
        return sum(numbers)

    return sum(numbers) / len(numbers)


def str2bool(v):
    """
    Maps string to boolean.
    :param v:
    :return:
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def clear_measurement_file(args):
    """
    Cleans the measurement file.
    :param args:
    :return:
    """
    open(get_output_path_performance_measurements(args), "w").close()


def set_seed(args):
    """
    Sets seed for reproducible results.
    :param args: args.
    :return: none.
    """
    np.random.seed(args.seed_val)
    tf.random.set_seed(args.seed_val)
    random.seed(args.seed_val)


def calculate_measures(args, _measures, predicted_distributions, ground_truths, best_model_id=0):
    prefix = 0
    prefix_all_enabled = 1
    predicted_label = list()
    ground_truth_label = list()

    output_path = get_output_path_predictions(args)

    with open(output_path, 'r') as result_file:
        result_reader = csv.reader(result_file, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        next(result_reader)

        for row in result_reader:
            if not row:
                continue
            else:
                if int(row[1]) == prefix or prefix_all_enabled == 1:
                    ground_truth_label.append(row[2])
                    predicted_label.append(row[3])

    _measures["accuracy_value"] = sklearn.metrics.accuracy_score(ground_truth_label, predicted_label)
    _measures["precision_micro_value"] = sklearn.metrics.precision_score(ground_truth_label, predicted_label,
                                                                         average='micro')
    _measures["precision_macro_value"] = sklearn.metrics.precision_score(ground_truth_label, predicted_label,
                                                                         average='macro')
    _measures["precision_weighted_value"] = sklearn.metrics.precision_score(ground_truth_label, predicted_label,
                                                                            average='weighted')
    _measures["recall_micro_value"] = sklearn.metrics.recall_score(ground_truth_label, predicted_label, average='micro')
    _measures["recall_macro_value"] = sklearn.metrics.recall_score(ground_truth_label, predicted_label, average='macro')
    _measures["recall_weighted_value"] = sklearn.metrics.recall_score(ground_truth_label, predicted_label,
                                                                      average='weighted')
    _measures["f1_micro_value"] = sklearn.metrics.f1_score(ground_truth_label, predicted_label, average='micro')
    _measures["f1_macro_value"] = sklearn.metrics.f1_score(ground_truth_label, predicted_label, average='macro')
    _measures["f1_weighted_value"] = sklearn.metrics.f1_score(ground_truth_label, predicted_label, average='weighted')
    _measures["f1_weighted_value"] = sklearn.metrics.f1_score(ground_truth_label, predicted_label, average='weighted')

    _measures["auc_roc_value"] = multi_class_roc_auc_score(args, ground_truths,
                                                           predicted_distributions, best_model_id=best_model_id)

    return _measures


def multi_class_roc_auc_score(args, ground_truths, prob_dist, average='macro', multi_class='ovr', best_model_id=0):
    """
    Calculate roc_auc_score
    Note:
    - multi-class ROC AUC currently only handles the ‘macro’ and ‘weighted’ averages.
    - this implementation can be used with binary, multi-class and
    multi-label classification, but some restrictions apply (see Parameters).
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
    - one-vs-rest ROC AUC scores
    We calculate the ROC AUC according to:
    Fawcett, T., 2006, "An introduction to ROC analysis. Pattern Recognition Letters", 27(8), pp. 861-874.
    Parameters
    ----------
    ground_truth_label : list of lists
        A sublist represents the encoding of a true activity (= label)
    predicted_label : list of lists
        A sublist represents the encoding of a predicted activity (= label)
    average : str
        The type of averaging performed on the data.
    Returns
    -------
    float :
        ROC AUC score
    """

    try:

        num_classes = len(prob_dist[0])
        num_instances = len(prob_dist)

        if args.encoding_cat == 'int':
            ground_truths_ = ground_truths
        else:
            ground_truths_ = [np.argmax(ground_truth) for ground_truth in ground_truths]

        ground_truths_unique = list(set(ground_truths_))
        ground_truths = np.asarray(ground_truths)
        prob_dist = np.asarray(prob_dist)

        prob_dist_existing_column = np.zeros((num_instances, len(ground_truths_unique)))
        ground_truths_existing_column = np.zeros((num_instances, len(ground_truths_unique)))

        if args.encoding_cat == 'int':

            # Create ground truth_id mapping
            mapped_gt_id = {}
            ground_truth_ids = []
            for idx, ground_truth in enumerate(ground_truths_unique):
                mapped_gt_id[ground_truth] = idx
                ground_truth_ids.append(idx)

            # Create list with class ids
            model = load_nap_model(args, best_model_id)
            model_classes = model.classes_  # get number of model classes from sklearn model
            model_classes_ids = []

            for class_ in model_classes:
                model_classes_ids.append(class_ - 1)

            # Set prob dist
            idx_existing_column = 0
            for idx_column in range(0, num_classes):
                if idx_column in model_classes_ids and idx_column in ground_truth_ids:  # if column does not include a probability
                    for idx_row in range(0, num_instances):
                        prob_dist_existing_column[idx_row, idx_existing_column] = prob_dist[idx_row, idx_column]
                    idx_existing_column += 1

            # Set ground truths
            for idx_row in range(0, num_instances):
                ground_truths_existing_column[idx_row, mapped_gt_id[ground_truths_[idx_row]]] = 1

        else:

            # set prob dist
            idx_existing_column = 0
            for idx_column in ground_truths_unique:
                for idx_row in range(0, num_instances):
                    prob_dist_existing_column[idx_row, idx_existing_column] = prob_dist[idx_row, idx_column]
                idx_existing_column += 1

            # set ground truths
            idx_existing_column = 0
            for idx_column in ground_truths_unique:
                for idx_row in range(0, num_instances):
                    ground_truths_existing_column[idx_row, idx_existing_column] = ground_truths[idx_row, idx_column]
                idx_existing_column += 1

            value = sklearn.metrics.roc_auc_score(ground_truths_existing_column, prob_dist_existing_column,
                                                  average=average,
                                                  multi_class=multi_class)
    except:
        value = 0

    return value


def print_measures(args, _measures):
    ll_print("\nAccuracy: %f\n" % (_measures["accuracy_value"]))

    ll_print("Precision (micro): %f\n" % (_measures["precision_micro_value"]))
    ll_print("Precision (macro): %f\n" % (_measures["precision_macro_value"]))
    ll_print("Precision (weighted): %f\n" % (_measures["precision_weighted_value"]))

    ll_print("Recall (micro): %f\n" % (_measures["recall_micro_value"]))
    ll_print("Recall (macro): %f\n" % (_measures["recall_macro_value"]))
    ll_print("Recall (weighted): %f\n" % (_measures["recall_weighted_value"]))

    ll_print("F1-Score (micro): %f\n" % (_measures["f1_micro_value"]))
    ll_print("F1-Score (macro): %f\n" % (_measures["f1_macro_value"]))
    ll_print("F1-Score (weighted): %f\n" % (_measures["f1_weighted_value"]))

    ll_print("AUC ROC Score: %f\n" % (_measures["auc_roc_value"]))

    if args.mode == 0:
        ll_print("Training time total: %f seconds\n" % (_measures["training_time_seconds"]))

        ll_print("Prediction time avg: %f seconds\n" % (avg(_measures["prediction_times_seconds"])))
        ll_print("Prediction time total: %f seconds\n" % (sum(_measures["prediction_times_seconds"])))

    if args.mode == 2:
        ll_print("Explanation time avg: %f seconds\n" % (avg(_measures["explanation_times_seconds"])))
        ll_print("Explanation time total: %f seconds\n" % (sum(_measures["explanation_times_seconds"])))
    ll_print("\n")


def write_measures(args, _measures):
    """
    Writes measures.
    :param args:
    :param _measures:
    :return:
    """

    names = ["dataset",
             "classifier",
             "model-id",
             "validation",
             "accuracy",
             "precision-micro",
             "precision-macro",
             "precision-weighted",
             "recall-micro",
             "recall-macro",
             "recall-weighted",
             "f1-score-micro",
             "f1-score-macro",
             "f1-score-weighted",
             "auc-roc"
             ]

    if args.mode == 0:
        names.extend(["training-time-total", "prediction-time-avg", "prediction-time-total"])

    if args.mode == 2:
        names.extend(["explainer", "removed_events_relevance", "removed_events_num", "explanation-time-avg",
                      "explanation-time-total"])

    names.append("time-stamp")

    model_id = args.model_id
    dataset = args.data_set[:-4]
    classifier = args.classifier
    mode = "split-%s" % args.split_rate_train

    values = [dataset, classifier, model_id, mode,
              _measures["accuracy_value"],
              _measures["precision_micro_value"],
              _measures["precision_macro_value"],
              _measures["precision_weighted_value"],
              _measures["recall_micro_value"],
              _measures["recall_macro_value"],
              _measures["recall_weighted_value"],
              _measures["f1_micro_value"],
              _measures["f1_macro_value"],
              _measures["f1_weighted_value"],
              _measures["auc_roc_value"]]

    if args.mode == 0:
        values.append(_measures["training_time_seconds"])
        values.append(avg(_measures["prediction_times_seconds"]))
        values.append(sum(_measures["prediction_times_seconds"]))

    values.append(arrow.now())

    output_path = get_output_path_performance_measurements(args)

    with open(output_path, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter=';', quoting=csv.QUOTE_NONE, escapechar=' ')
        if os.stat(output_path).st_size == 0:
            # If file is empty
            writer.writerow(names)
        writer.writerow(values)


def get_output_path_performance_measurements(args):
    directory = './%s%s' % (args.task, args.result_dir[1:])

    if args.mode == 0:
        file = 'measures_%s_%s.csv' % (args.data_set[:-4], args.classifier)
    if args.mode == 2:
        file = 'measures_%s_%s_manipulated.csv' % (args.data_set[:-4], args.classifier)

    return directory + file


def get_output_path_predictions(args):
    directory = './' + args.task + args.result_dir[1:]
    file = args.data_set.split(".csv")[0]

    file += "_0_%s" % args.classifier

    if args.mode == 2:
        file += "_manipulated"
    file += ".csv"

    return directory + file


def get_model_dir(args, model_id=0):
    """
    Returns the path to the stored trained model for the next activity prediction.

    Parameters
    ----------
    args : Namespace
        Settings of the configuration parameters.
    model_id : int
        ID of a model saved during hyperparameter optimization.

    Returns
    -------
    str :
        Path to stored model.

    """
    model_dir = "%s%s_%s_%s_%s" % (args.model_dir, args.task,
                                   args.data_set[0:len(args.data_set) - 4], args.classifier, args.model_id)
    if args.hpo:
        model_dir += "_trial%s" % model_id
    if args.classifier == "LSTM":
        model_dir += ".h5"
    else:
        model_dir += ".joblib"

    return model_dir


def load_nap_model(args, model_id=0):
    """
    Returns ML model used for next activity prediction.

    Parameters
    ----------
    args : Namespace
        Settings of the configuration parameters.
    model_id : int
        ID of a model saved during hyperparameter optimization.

    Returns
    -------
    model : type depends on classifier type

    """

    model_dir = get_model_dir(args, model_id)
    if args.classifier == "LSTM":
        model = load_model(model_dir)
    else:
        model = load(model_dir)

    return model


def add_to_file(args, file_type, trail):
    file_name = './%s%s%s_%s_%s.csv' % (args.task, args.result_dir[1:],
                                        file_type, args.data_set[:-4], args.classifier)

    file = open(file_name, "a+")

    if file_type == "hyper_params":

        if os.stat(file_name).st_size != 0:
            file.write("\n")

        file.write("Model id: " + str(args.model_id) + "\n")
        file.write("Dataset: " + str(args.data_set[:-4]) + "\n")
        file.write("Best accuracy value: " + str(round(trail.value, 4)) + "\n")
        file.write("Encoding: " + str(args.encoding_cat) + "\n")
        file.write("Best params:\n")

        for key, value in trail.params.items():
            file.write("%s: %s\n" % (str(key), str(value)))
        file.write("--------------------------\n")

    file.close()
