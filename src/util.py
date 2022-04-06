import numpy as np


def get_one_hot_of_activity_mimic(x):
    if x['Activity'] == 'PHYS REFERRAL/NORMAL DELI':
        ret = [0, 1]  # No additional information, so normal one hot
    elif x['Activity'] == 'HOME':
        ret = [1, 1]  # No additional information, so normal one hot
    elif x['Activity'] == 'EMERGENCY ROOM ADMIT':
        ret = [2, 1]  # No additional information, so normal one hot
    elif x['Activity'] == 'SNF':
        ret = [3, 1]  # No additional information, so normal one hot
    elif x['Activity'] == 'HOME WITH HOME IV PROVIDR':
        ret = [4, 1]  # No additional information, so normal one hot
    elif x['Activity'] == 'HOME HEALTH CARE':
        ret = [5, 1]  # No additional information, so normal one hot
    elif x['Activity'] == 'DEAD/EXPIRED':
        ret = [6, 1]  # No additional information, so normal one hot
    elif x['Activity'] == 'SHORT TERM HOSPITAL':
        ret = [7, 1]  # No additional information, so normal one hot
    elif x['Activity'] == 'TRANSFER FROM HOSPEXTRAM':
        ret = [8, 1]  # No additional information, so normal one hot
    elif x['Activity'] == 'REHABDISTINCT PART HOSP':
        ret = [9, 1]  # No additional information, so normal one hot
    elif x['Activity'] == 'DISC-TRAN CANCER/CHLDRN H':
        ret = [10, 1]  # No additional information, so normal one hot
    elif x['Activity'] == 'CLINIC REFERRAL/PREMATURE':
        ret = [11, 1]  # No additional information, so normal one hot
    elif x['Activity'] == 'LONG TERM CARE HOSPITAL':
        ret = [12, 1]  # No additional information, so normal one hot
    elif x['Activity'] == 'DISC-TRAN TO FEDERAL HC':
        ret = [13, 1]  # No additional information, so normal one hot
    elif x['Activity'] == 'HOSPICE-MEDICAL FACILITY':
        ret = [14, 1]  # No additional information, so normal one hot
    elif x['Activity'] == 'LEFT AGAINST MEDICAL ADVI':
        ret = [15, 1]  # No additional information, so normal one hot
    elif x['Activity'] == 'HOSPICE-HOME':
        ret = [16, 1]  # No additional information, so normal one hot
    elif x['Activity'] == 'TRANSFER FROM OTHER HEALT':
        ret = [17, 1]  # No additional information, so normal one hot
    elif x['Activity'] == 'DISCH-TRAN TO PSYCH HOSP':
        ret = [18, 1]  # No additional information, so normal one hot
    elif x['Activity'] == 'TRANSFER FROM SKILLED NUR':
        ret = [19, 1]  # No additional information, so normal one hot
    elif x['Activity'] == 'HMO REFERRAL/SICK':
        ret = [20, 1]  # No additional information, so normal one hot
    elif x['Activity'] == '** INFO NOT AVAILABLE **':
        ret = [21, 1]  # No additional information, so normal one hot
    elif x['Activity'] == 'OTHER FACILITY':
        ret = [22, 1]  # No additional information, so normal one hot
    elif x['Activity'] == 'ICF':
        ret = [23, 1]  # No additional information, so normal one hot
    elif x['Activity'] == 'SNF-MEDICAID ONLY CERTIF':  # todo??
        ret = [24, 1]  # No additional information, so normal one hot
    elif x['Activity'] == 'TRSF WITHIN THIS FACILITY':
        ret = [25, 1]  # No additional information, so normal one hot

    one_hot = np.zeros(26, dtype=np.float32)

    one_hot[ret[0]] = ret[1]

    return one_hot


def get_one_hot_of_activity_sim(x, max_lacticacid, max_crp):

    if x['Activity'] == 'Start':
        ret = [0, 1]  # No additional information, so normal one hot
    elif x['Activity'] == 'IVL':
        ret = [1, 1]  # No additional information, so normal one hot
    elif x['Activity'] == 'IVA':
        ret = [2, 1]  # No additional information, so normal one hot
    if x['Activity'] == 'CRP':
        ret = [3, min(x['CRP'], max_crp) / max_crp]
        if np.isnan(ret[1]):
            ret[1] = -1
    elif x['Activity'] == 'LacticAcid':
        ret = [4, min(x['LacticAcid'], max_lacticacid) / max_lacticacid]
        if np.isnan(ret[1]):
            ret[1] = -1
    one_hot = np.zeros(5, dtype=np.float32)
    one_hot[ret[0]] = ret[1]

    return one_hot


def get_one_hot_of_activity_sepsis(x, max_leucocytes, max_lacticacid, max_crp):
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

    return one_hot
