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
import random


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

def concatenate_tensor_matrix(x_seq, x_stat):
    x_train_seq_ = x_seq.reshape(-1, x_seq.shape[1] * x_seq.shape[2])
    x_concat = np.concatenate((x_train_seq_, x_stat), axis=1)

    return x_concat

def read_data(dir=r"..\data_plot\test_data"):
    data_list = []
    with open(dir, "rb") as input:
        while True:
            try:
                x = pickle.load(input)
            except EOFError:
                break
            data_list.append(x)

    return data_list


def read_models(seed, dir=r"..\model"):
    model_list = []
    for file in os.listdir(dir):
        filename = os.fsdecode(file)
        if str(seed) in filename:
            model_list.append(torch.load(dir + "\\" + file))

    return model_list


def get_prefix_length(seq, sample):
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


def get_prefix_dictionary(seq):
    samples = seq.shape[0]
    print(samples)

    mapList = []
    uniquePrefixSizes = []
    for t in range(0, samples):
        indexPrefixSizeMap = {"index": t, "prefixSize": get_prefix_length(seq, t)}
        # print("sample " + str(t) + ": " + str(getPrefixLength(seq, t)))
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

    # print(prefixDictionary)
    return prefixDictionaryList


def get_plot_data(model_list, data_list):
    result = []
    for model, dataset in zip(model_list, data_list):
        values = {"model": dataset["fold"], "data": []}

        seq = torch.from_numpy(dataset["x_test_seq"])
        stat = torch.from_numpy(dataset["x_test_stat"])
        label = torch.from_numpy(dataset["label"]).numpy()

        prefixLengthDictList = get_prefix_dictionary(seq)

        for prefixLengthDict in prefixLengthDictList:
            performancePrefixDict = {"PrefixLength": prefixLengthDict["prefixSize"]}

            prefixSeq = seq[prefixLengthDict["indizes"], :, :]
            prefixStat = stat[prefixLengthDict["indizes"], :]
            prefixLabel = label[prefixLengthDict["indizes"]]

            model.eval()
            with torch.no_grad():
                prediction = torch.sigmoid(model.forward(prefixSeq, prefixStat))

            prediction = prediction.numpy()
            performancePrefixDict["AUC"] = calc_roc_auc(prefixLabel, prediction)
            values["data"].append(performancePrefixDict)

        result.append(values)

    return result


def get_average_result(result, conf: bool = False, label=""):
    # ax = plt.subplot
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

    avg_y = sum(avg_y) / 5
    avg_y = avg_y.fillna(0)

    return {"x": avg_x, "y": avg_y, "conf": conf, "label": label}


def plot_data(data):
    '''
    :param data: a list of dictionaries returned from get_average_result()
    :return: a plot with all plots stored in the data list in it
    '''
    fig = plt.figure()

    for entry in data:
        if entry["conf"]:
            plt.plot(entry["x"], entry["y"], c="cyan", label=entry["label"], marker="o", ls="--")
            confiI = 0.05 * np.std(entry["y"]) / np.mean(entry["y"])
            plt.fill_between(entry["x"], (entry["y"] - confiI), (entry["y"] + confiI), color="cyan", alpha=0.1)
        else:
            b = random.uniform(0, 0.5)
            col = (np.random.random(), np.random.random(), b)

            plt.plot(entry["x"], entry["y"], c=col, label=entry["label"], marker="o", ls="--")

    plt.xlabel("Prefix Length")
    plt.ylabel("ROC AUC")
    plt.xticks(range(0, 25))
    plt.legend()

    return fig

def save_procedure_plot_data(dir = r"..\data_plot\test_data"):
    r_data = read_data()
    r_models = read_models(r_data[0]["seed"]) #Der Seed ist in allen Einträgen der r_data Liste gleich, daher kann man einfach Index 0 nehmen

    open(dir, 'w').close() #Das File muss geleert werden, damit Daten von anderen Verfahren einlesen werden können

    conf = False
    if r_data[0]["procedure"] == "pwn": #Das Vorgehen ist in allen Einträgen der r_data Liste gleich, daher kann man einfach Index 0 nehmen
        conf = True

    with open(r"..\data_plot\plot_data", "ab") as output:
        pickle.dump(get_average_result(get_plot_data(r_models, r_data), conf, label= r_data[0]["procedure"]), output)

    #return get_average_result(get_plot_data(r_models, r_data), conf, label= r_data["procedure"])


def plot_everything_saved():
    plot_list = []
    with open(r"..\data_plot\plot_data", "rb") as input:
        while True:
            try:
                x = pickle.load(input)
            except EOFError:
                break
            plot_list.append(x)

    plot_data(plot_list)
    plt.show()

def clear_plot_data_file():
    open(r"..\data_plot\plot_data", 'w').close()

'''
x = get_average_result(get_plot_data(read_models(), read_data()), True, label="Methode 1")
y = get_average_result(get_plot_data(read_models(), read_data()), label="Methode 2")
z = get_average_result(get_plot_data(read_models(), read_data()), label="Methode 3")
y["y"] *= 2
z["y"] += .3
z["y"] *= 0.5
plot = plot_data([x, y, z])
# combined_fig = combine_figures([x,y])
plt.show()
'''

#plot_everything_saved()
#plt.show()
'''
plot_list = []
with open(r"..\data_plot\plot_data", "rb") as input:
    while True:
        try:
            x = pickle.load(input)
        except EOFError:
            break
        plot_list.append(x)

print(plot_list[1]["conf"])
'''