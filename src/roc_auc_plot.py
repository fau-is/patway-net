import pickle
import pandas as pd
import numpy as np
import torch
import os
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt


# Straight up stolen from main.py
def calc_roc_auc(gts, probs):
    try:
        auc = metrics.roc_auc_score(gts, probs)
        if np.isnan(auc):
            auc = 0
        return auc
    except:
        return 0

def read_data(dir=f"../data_plot/test_data"):
    '''
    Reads dataset-dictionaries saved in main.py from file
    :param dir: path of the file
    :return: dataset-dictionaries in a list
    '''

    data_list = []
    with open(dir, "rb") as input:
        while True:
            try:
                x = pickle.load(input)
            except EOFError:
                break
            data_list.append(x)

    return data_list


def read_models(seed, mode, dir=r"..\model"):
    '''
    Reads ml models saved in main.py from folder
    :param seed: seed of the saved models, important for identifying the models in the folder
    :param dir: path of the folder containing the models
    :return: dataset-dictionaries in a list
    '''

    model_list = []
    for file in os.listdir(dir):
        filename = os.fsdecode(file)
        if str(seed) in filename and mode in filename:
            model_list.append(torch.load(dir + "\\" + file))

    return model_list


def get_prefix_length(seq, sample):
    '''
    Determines the prefix-length of a given sample from a given sequentiell dataset
    :param seq: sequentiell dataset
    :param sample: sample of the sequentiell dataset which prefix-length need to be determined
    :return: prefix-length
    '''
    counter = 0
    row_counter = -1
    last_row = None
    actual_prefix_rows = 0
    for matrix in seq[sample, :, :]:

        row_counter += 1
        not_zero = False

        for x in matrix:
            x = float(x)
            if x != 0:
                not_zero = True
                break

        if not_zero:
            last_row = matrix
            actual_prefix_rows = row_counter

    if last_row != None:
        for entry in last_row:
            if entry == 0:
                counter += 1

            if entry != 0:
                counter += 1
                break

    # return counter + (actual_prefix_rows * len(last_row))  #Every value counts into the prefix size
    return actual_prefix_rows + 1  # Every row counts into the prefix size


def get_prefix_dictionary(seq, max_prefix_size=15):
    '''
    Creates a list of dictionaries mapping all samples of a sequentiell dataset with their prefix-length
    :param seq: sequential dataset
    :param max_prefix_size: maximal prefix length documented by the returned dictionary list
    :return: list containing dictionaries mapping all samples of a sequential dataset with their prefix length
    '''
    samples = seq.shape[0]

    map_list = []
    unique_prefix_sizes = []
    for t in range(0, samples):
        index_prefix_size_map = {"index": t, "prefix_size": get_prefix_length(seq, t)}

        map_list.append(index_prefix_size_map)
        unique_prefix_sizes.append(get_prefix_length(seq, t))

    unique_prefix_sizes = set(unique_prefix_sizes)

    prefix_dictionary_list = []

    for prefix_size in unique_prefix_sizes:
        prefix_dictionary = {}
        prefix_dictionary["prefix_size"] = prefix_size
        prefix_dictionary["indizes"] = []
        for entry in map_list:
            if entry["prefix_size"] == prefix_size:
                prefix_dictionary["indizes"].append(entry["index"])
        prefix_dictionary_list.append(prefix_dictionary)

    filter_list = []
    for entry in prefix_dictionary_list:
        if entry["prefix_size"] <= max_prefix_size:
            filter_list.append(entry)

    return filter_list


def get_plot_data(model_list, data_list, max_prefix_size, model_name):
    '''
    Creates data that is needed from the get_average_result() function to create plot data
    :param model_list: ml models
    :param data_list: list of datasets
    :param max_prefix_size: maximal prefix length used for plotting
    :param model: name of model
    :return: list of dictionaries containing the fold and a dictionary containing the prefix-length and the according AUC
    '''
    max_prefix_size += 1

    result = []
    for model, dataset in zip(model_list, data_list):

        values = {"model": dataset["fold"], "data": []}

        seq = torch.from_numpy(dataset["x_test_seq"])
        stat = torch.from_numpy(dataset["x_test_stat"])
        label = torch.from_numpy(dataset["label"]).numpy()

        prefix_length_dict_list = get_prefix_dictionary(seq, max_prefix_size)

        for prefix_length_dict in prefix_length_dict_list:
            performance_prefix_dict = {"PrefixLength": prefix_length_dict["prefixSize"]}

            prefix_seq = seq[prefix_length_dict["indizes"], :, :]
            prefix_stat = stat[prefix_length_dict["indizes"], :]
            prefix_label = label[prefix_length_dict["indizes"]]

            if model_name == "pwn_one_inter" or model_name == "lstm" or model_name == "pwn_no_inter":
                model.eval()
                with torch.no_grad():
                    preds_proba = torch.sigmoid(model.forward(prefix_seq, prefix_stat))
                    preds_proba = [pred_proba[0] for pred_proba in preds_proba]
            else:
                preds_proba = model.predict_proba(prefix_stat)
                preds_proba = [pred_proba[1] for pred_proba in preds_proba]

            performance_prefix_dict["AUC"] = calc_roc_auc(prefix_label, preds_proba)
            values["data"].append(performance_prefix_dict)

        result.append(values)

    return result


def get_average_result(result, conf: bool = False, label=""):
    '''
    Creates plot-data for the average result over all folds.
    :param result: input data created by get_plot_data()
    :param conf: bool deciding if the line has a confidence intervall or not
    :param label: label of the line
    :return: a dictionary containing, x- and y-values, a bool, if a confidence intervall will be plottet for this line and a label
    '''

    avg_y = []
    avg_x = []
    for values in result:

        data = values["data"]
        df = pd.DataFrame(columns=["prefix_length", "AUC"])
        for i in range(0, len(data) - 1):
            x = (data[i]["prefix_length"])
            y = (data[i]["AUC"])
            df.loc[i] = (x, y)

        df = df.sort_values(by=["prefix_length"])
        avg_y.append(df["AUC"])

        if len(df["prefix_length"]) > len(avg_x):
            avg_x = df["prefix_length"]

    avg_y = sum(avg_y) / len(avg_y)
    avg_y = avg_y.fillna(0)

    print(f"x: {str([x for x in avg_x])},\n y: {str([y for y in avg_y])}, \n conf: {conf},\n label: {label}\n\n")

    return {"x": avg_x, "y": avg_y, "conf": conf, "label": label}


def plot_data(results, max_prefix_size=15):
    '''
    :param results: a list of dictionaries returned from get_average_result()
    :param max_prefix_size: maximal prefix length used for plotting
    :return: a plot with all plots stored in the data list in it
    '''

    fig = plt.figure()
    palet = sns.color_palette()

    for i, result in enumerate(results):
        result["x"] = np.array(result["x"], dtype=float)
        result["y"] = np.array(result["y"], dtype=float)

        print(result["x"][0])

        if result["conf"]:
            plt.plot(result["x"], result["y"], c="cyan", label=result["label"], marker="o", ls="--")
            confiI = 1.96 * np.std(result["y"]) / np.sqrt(len(result["y"]))
            plt.fill_between(result["x"], (result["y"] - confiI), (result["y"] + confiI), color="cyan", alpha=0.2)
        else:
            plt.plot(result["x"], result["y"], label=result["label"], marker="o", ls="--", color=palet[i])

    plt.xlabel("Prefix Length")
    plt.ylabel("ROC AUC")

    tick_range = [x + 1 for x in range(0, max_prefix_size)]
    plt.xticks(tick_range)

    plt.legend()

    return fig

def plot_everything_saved(max_prefix_size=15, save=False):
    '''
    Plots everything that was saved by save_procedure_plot_data()
    :param max_prefix_size: maximal prefix length used for plotting
    '''

    plot_list = []
    with open(f"../data_prediction_plot/plot_data", "rb") as input:
        while True:
            try:
                x = pickle.load(input)
            except EOFError:
                break
            plot_list.append(x)

    plot_data(plot_list, max_prefix_size)

    if save:
        name = str(dt.datetime.now())
        name = name.replace(".", "-")
        name = name.replace(":", "-")
        plt.savefig(r"..\plots\prediction" + "\\" + name + ".png")
        plt.show()


def clear_plot_data_file():
    '''
    Clears the plot_data file.
    '''

    open(f"../data_prediction_plot/plot_data", 'w').close()


if __name__ == "__main__":

    clear_plot_data_file()

    seed = 245
    dir_pairs = [(f"../data_prediction_plot/test_data_{seed}", f"../model")]
    max_prefix_size = 30
    model_names = ["pwn_one_inter", "pwn_no_inter", "lstm", "lr", "dt", "knn", "nb"]
    model_names_paper = ["PatWay-Net (with interaction)",
                         "PatWay-Net (without interaction)",
                         "LSTM network (with static module)",
                         "Logistic regression", "Decision tree",
                         "K-nearest neighbor", "Naive Bayes"]

    for pair in dir_pairs:
        for i, model_name in enumerate(model_names):
            data_list = read_data(pair[0])
            model_list = read_models(seed, model_name, pair[1])

            conf = False
            if model_name == "pwn_one_inter":
                conf = True

            with open(r"..\data_prediction_plot\plot_data", "ab") as output:
                pickle.dump(get_average_result(get_plot_data(model_list, data_list, max_prefix_size, model_name),
                                               conf, label=model_names_paper[i]), output)

    plot_everything_saved(max_prefix_size, save=True)
