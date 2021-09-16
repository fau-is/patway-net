import numpy as np


def get_custom_one_hot_of_activity(x, max_leucocytes, max_lacticacid):
    if x['Activity'] == 'Leucocytes':
        ret = [0, min(x['Leucocytes'], max_leucocytes) / max_leucocytes]
        if np.isnan(ret[1]):
            ret[1] = -1
    elif x['Activity'] == 'CRP':
        ret = [1, x['CRP']]
        if np.isnan(ret[1]):
            ret[1] = -1
    elif x['Activity'] == 'LacticAcid':
        ret = [2, min(x['LacticAcid'], max_lacticacid) / max_lacticacid]
        if np.isnan(ret[1]):
            ret[1] = -1
    elif x['Activity'] == 'ER Registration':
        ret = [3, 1]  # No additional information, so normal one hot
    elif x['Activity'] == 'ER Triage':
        ret = [4, 1]  # No additional information, so normal one hot
    elif x['Activity'] == 'ER Sepsis Triage':
        ret = [5, 1]  # No additional information, so normal one hot
    elif x['Activity'] == 'IV Liquid':
        ret = [6, 1]  # No additional information, so normal one hot
    elif x['Activity'] == 'IV Antibiotics':
        ret = [7, 1]  # No additional information, so normal one hot
    elif x['Activity'] == 'Admission NC':
        ret = [8, 1]  # No additional information, so normal one hot
    elif x['Activity'] == 'Admission IC':
        ret = [9, 1]  # No additional information, so normal one hot
    elif x['Activity'] == 'Return ER':
        ret = [10, 1]  # No additional information, so normal one hot
    elif x['Activity'] == 'Release A':
        ret = [11, 1]  # No additional information, so normal one hot
    elif x['Activity'] == 'Release B':
        ret = [12, 1]  # No additional information, so normal one hot
    elif x['Activity'] == 'Release C':
        ret = [13, 1]  # No additional information, so normal one hot
    elif x['Activity'] == 'Release D':
        ret = [14, 1]  # No additional information, so normal one hot
    elif x['Activity'] == 'Release E':
        ret = [15, 1]  # No additional information, so normal one hot
    one_hot = np.zeros(16, dtype=np.float32)
    one_hot[ret[0]] = ret[1]

    return one_hot
