import pickle
import numpy
import pandas as pd
import numpy as np
import os
import src.data as data
import torch
import os
from sklearn import metrics
import matplotlib.pyplot as plt
import math
import seaborn as sns
import random
import datetime as dt


# Straight up stolen from main.py
def calc_roc_auc(gts, probs):
    try:
        auc = metrics.roc_auc_score(gts, probs)
        if np.isnan(auc):
            auc = 0
        return auc
    except:
        print("except")
        return 0


# Straight up stolen from main.py
def concatenate_tensor_matrix(x_seq, x_stat):
    x_train_seq_ = x_seq.reshape(-1, x_seq.shape[1] * x_seq.shape[2])
    x_concat = np.concatenate((x_train_seq_, x_stat), axis=1)

    return x_concat


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
    rowCounter = -1
    lastRow = None
    actualPrefixRows = 0
    for matrix in seq[sample, :, :]:
        # print(matrix)
        rowCounter += 1

        notZero = False

        for x in matrix:
            x = float(x)
            if x != 0:
                notZero = True
                break

        if notZero:
            lastRow = matrix
            actualPrefixRows = rowCounter

    if lastRow != None:
        for entry in lastRow:
            if entry == 0:
                counter += 1

            if entry != 0:
                counter += 1
                break

    # print(actualPrefixRows)
    # print(counter)
    # print(rowCounter)
    # print(lastRow)

    # return counter + (actualPrefixRows * len(lastRow))  #Every Value counts into the prefix size
    return actualPrefixRows + 1  # Every Row counts into the prefix size


def get_prefix_dictionary(seq, max_prefix_size=15):
    '''
    Creates a list of dictionaries mapping all samples of a sequentiell dataset with their prefix-length
    :param seq: sequential dataset
    :param max_prefix_size: maximal prefix length documented by the returned dictionary list
    :return: list containing dictionaries mapping all samples of a sequential dataset with their prefix length
    '''
    samples = seq.shape[0]
    print(samples)

    mapList = []
    uniquePrefixSizes = []
    for t in range(0, samples):
        indexPrefixSizeMap = {"index": t, "prefixSize": get_prefix_length(seq, t)}

        mapList.append(indexPrefixSizeMap)
        uniquePrefixSizes.append(get_prefix_length(seq, t))

    uniquePrefixSizes = set(uniquePrefixSizes)
    # print(uniquePrefixSizes)

    prefixDictionaryList = []

    for prefixSize in uniquePrefixSizes:
        prefixDictionary = {}
        prefixDictionary["prefixSize"] = prefixSize
        prefixDictionary["indizes"] = []
        for entry in mapList:
            if entry["prefixSize"] == prefixSize:
                prefixDictionary["indizes"].append(entry["index"])
        prefixDictionaryList.append(prefixDictionary)

    filter_list = []
    for entry in prefixDictionaryList:
        if entry["prefixSize"] <= max_prefix_size:
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

            if model_name == "pwn" or model_name == "lstm" or model_name == "pwn_no_inter":
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
        df = pd.DataFrame(columns=["PrefixLength", "AUC"])
        for i in range(0, len(data) - 1):
            x = (data[i]["PrefixLength"])
            y = (data[i]["AUC"])
            df.loc[i] = (x, y)

        df = df.sort_values(by=["PrefixLength"])
        avg_y.append(df["AUC"])

        if len(df["PrefixLength"]) > len(avg_x):
            avg_x = df["PrefixLength"]

    avg_y = sum(avg_y) / len(avg_y)
    avg_y = avg_y.fillna(0)

    return {"x": avg_x, "y": avg_y, "conf": conf, "label": label}


def plot_data(results, max_prefix_size=15):
    '''
    :param results: a list of dictionaries returned from get_average_result()
    :param max_prefix_size: maximal prefix length used for plotting
    :return: a plot with all plots stored in the data list in it
    '''

    fig = plt.figure()
    ax = plt.gca()
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
            # b = random.uniform(0, 0.5)
            # col = (np.random.random(), np.random.random(), b)

            plt.plot(result["x"], result["y"], label=result["label"], marker="o", ls="--", color=palet[i])

    plt.xlabel("Prefix Length")
    plt.ylabel("ROC AUC")

    tick_range = [x + 1 for x in range(0, max_prefix_size)]
    plt.xticks(tick_range)

    plt.legend()

    return fig

"""
def save_procedure_plot_data(max_prefix_size=15, dir=f"../data_prediction_plot/test_data"):
    '''
    Creates and saves plot data for currently used ml procedure in main.py. Data will be saved to a file
    :param dir: path of the file where the datasets from the current procedure where saved (main.py)
    '''
    r_data = read_data()
    r_models = read_models(r_data[0]["seed"])  # the seed of the r_data list is the same for every entry, therefore I use the seed from index 0
    

    open(dir, 'w').close()  # the file needs to be cleared, so data from another procedure can be saved

    conf = False
    if r_data[0]["procedure"] == "pwn":  # the procedure of the r_data list is the same for every entry, therefore I use the seed from index 0
        conf = True

    with open(f"../data_prediction_plot/plot_data", "ab") as output:
        pickle.dump(get_average_result(get_plot_data(
            r_models, r_data, max_prefix_size), conf, label=r_data[0]["procedure"]), output)
"""

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
    # plt.rcParams['text.usetex'] = True
    # plt.style.use('science')

    seed = 137
    dir_pairs = [(f"../data_prediction_plot/test_data_{seed}", f"../model")]
    max_prefix_size = 15
    model_names = ["pwn", "pwn_no_inter", "lstm", "lr", "dt", "knn", "nb"]
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
            if model_name == "pwn":
                conf = True

            with open(r"..\data_prediction_plot\plot_data", "ab") as output:
                pickle.dump(get_average_result(get_plot_data(model_list, data_list,
                                                             max_prefix_size, model_name),
                                               conf, label=model_names_paper[i]), output)

    plot_everything_saved(max_prefix_size, save=True)
