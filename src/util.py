import numpy as np


def get_one_hot_of_activity_sim(x, max_blood_pressure, max_heart_rate, current_blood_pressure_value, current_heart_rate_value):
    if x['Activity'] == 'ER Registration':
        ret = [0, 1]  # No additional information, so normal one hot
    elif x['Activity'] == 'Medication B':
        ret = [1, 1]  # No additional information, so normal one hot
    elif x['Activity'] == 'Medication A':
        ret = [2, 1]  # No additional information, so normal one hot
    if x['Activity'] == 'Heart Rate':
        ret = [3, min(x['Heart Rate'], max_heart_rate) / max_heart_rate]
        if np.isnan(ret[1]):
            ret[1] = -1
    elif x['Activity'] == 'Blood Pressure':
        ret = [4, min(x['Blood Pressure'], max_blood_pressure) / max_blood_pressure]
        if np.isnan(ret[1]):
            ret[1] = -1

    one_hot = np.zeros(5, dtype=np.float32)
    one_hot[ret[0]] = ret[1]

    # Set last value of seq features
    if np.isnan(x['Heart Rate']):
        one_hot[3] = current_heart_rate_value
    if np.isnan(x['Blood Pressure']):
        one_hot[4] = current_blood_pressure_value

    return one_hot, one_hot[3], one_hot[4]


def get_one_hot_of_activity_sepsis(x, max_leucocytes, max_crp, max_lacticacid, current_leucocytes_value,
                                   current_crp_value, current_lacticacid_value):

    if x['Activity'] == 'Leucocytes':
        ret = [0, min(x['Leucocytes'], max_leucocytes) / max_leucocytes]
        if np.isnan(ret[1]):
            ret[1] = -1
    elif x['Activity'] == 'CRP':
        ret = [1, min(x['CRP'], max_crp) / max_crp]
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

    one_hot[0] = min(x['Leucocytes'], max_leucocytes) / max_leucocytes
    one_hot[1] = min(x['CRP'], max_crp) / max_crp
    one_hot[2] = min(x['LacticAcid'], max_lacticacid) / max_lacticacid

    """
    # Set last value of seq features
    if np.isnan(x['Leucocytes']):
        one_hot[0] = current_leucocytes_value
    if np.isnan(x['CRP']):
        one_hot[1] = current_crp_value
    if np.isnan(x['LacticAcid']):
        one_hot[2] = current_lacticacid_value
    """

    return one_hot, one_hot[0], one_hot[1], one_hot[2]


def get_only_seq_measures_sepsis(x, max_leucocytes, max_crp, max_lacticacid, current_leucocytes_value,
                                   current_crp_value, current_lacticacid_value):
    ret = [0, 0]

    if x['Activity'] == 'Leucocytes':
        ret = [0, min(x['Leucocytes'], max_leucocytes) / max_leucocytes]
        if np.isnan(ret[1]):
            ret[1] = -1
    elif x['Activity'] == 'CRP':
        ret = [1, min(x['CRP'], max_crp) / max_crp]
        if np.isnan(ret[1]):
            ret[1] = -1
    elif x['Activity'] == 'LacticAcid':
        ret = [2, min(x['LacticAcid'], max_lacticacid) / max_lacticacid]
        if np.isnan(ret[1]):
            ret[1] = -1

    one_hot = np.zeros(3, dtype=np.float32)
    one_hot[ret[0]] = ret[1]

    # Set last value of seq features
    if np.isnan(x['Leucocytes']):
        one_hot[0] = current_leucocytes_value
    if np.isnan(x['CRP']):
        one_hot[1] = current_crp_value
    if np.isnan(x['LacticAcid']):
        one_hot[2] = current_lacticacid_value

    return one_hot, one_hot[0], one_hot[1], one_hot[2]