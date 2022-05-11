import pandas as pd
import src.util as util
import numpy as np


def get_sim_data(label, file):
    ds_path = f'../data/{file}'

    static_features = ['Gender', 'Foreigner', 'BMI', 'Age']
    seq_features = ['Start', 'IVL', 'IVA', 'CRP', 'LacticAcid']

    df = pd.read_csv(ds_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    age_max = max(df['Age'])
    df['Age'] = df['Age'].apply(lambda x: x / age_max)
    bmi_max = max(df['BMI'])
    df['BMI'] = df['BMI'].apply(lambda x: x / bmi_max)

    max_lacticacid = np.percentile(df['LacticAcid'].dropna(), 100)  # remove outliers
    max_crp = np.percentile(df['CRP'].dropna(), 100)  # remove outliers

    x_seqs = []
    x_statics = []
    x_time_vals = []
    y = []

    for case in df['Case ID'].unique():

        after_registration_flag = False
        df_tmp = df[df['Case ID'] == case]
        idx = -1
        current_crp_value = 0
        current_lacticacid_value = 0

        for _, x in df_tmp.iterrows():
            idx = idx + 1
            if x['Activity'] == 'Start' and idx == 0:
                x_statics.append(x[static_features].values.astype(float))
                x_time_vals.append([])
                x_seqs.append([])
                after_registration_flag = True

            if after_registration_flag:

                one_hot, current_crp_value, current_lacticacid_value = util.get_one_hot_of_activity_sim(x, max_lacticacid, max_crp, current_crp_value, current_lacticacid_value)
                x_seqs[-1].append(one_hot)
                x_time_vals[-1].append(x['Timestamp'])

        if after_registration_flag:
            y.append(x[label])

    assert len(x_seqs) == len(x_statics) == len(y) == len(x_time_vals)

    return x_seqs, x_statics, y, x_time_vals, seq_features, static_features



def get_sepsis_data(target_activity, max_len, min_len):
    ds_path = '../data/Sepsis Cases - Event Log_sub.csv'

    static_features = ['InfectionSuspected', 'DiagnosticBlood', 'DisfuncOrg',
                       'SIRSCritTachypnea', 'Hypotensie',
                       'SIRSCritHeartRate', 'Infusion', 'DiagnosticArtAstrup', 'Age',
                       'DiagnosticIC', 'DiagnosticSputum', 'DiagnosticLiquor',
                       'DiagnosticOther', 'SIRSCriteria2OrMore', 'DiagnosticXthorax',
                       'SIRSCritTemperature', 'DiagnosticUrinaryCulture', 'SIRSCritLeucos',
                       'Oligurie', 'DiagnosticLacticAcid', 'Diagnose', 'Hypoxie',
                       'DiagnosticUrinarySediment', 'DiagnosticECG']

    seq_features = ['Leucocytes', 'CRP', 'LacticAcid', 'ER Registration', 'ER Triage', 'ER Sepsis Triage',
                    'IV Liquid', 'IV Antibiotics', 'Admission NC', 'Admission IC',
                    'Return ER', 'Release A', 'Release B', 'Release C', 'Release D',
                    'Release E']

    int2act = dict(zip(range(len(seq_features)), seq_features))

    df = pd.read_csv(ds_path)
    df['Complete Timestamp'] = pd.to_datetime(df['Complete Timestamp'])

    # Sort case id by timestamp of first event
    df_ = df.groupby('Case ID').first()
    df_ = df_.sort_values(by='Complete Timestamp')
    x = pd.CategoricalDtype(df_.index.values, ordered=True)
    df['Case ID'] = df['Case ID'].astype(x)
    df = df.sort_values(['Case ID', 'Complete Timestamp'])
    df = df.reset_index()

    diagnose_mapping = dict(zip(df['Diagnose'].unique(), np.arange(len(df['Diagnose'].unique()))))  # ordinal encoding
    df['Diagnose'] = df['Diagnose'].apply(lambda x: diagnose_mapping[x])
    df['Diagnose'] = df['Diagnose'].apply(lambda x: x / max(df['Diagnose']))  # normalise ordinal encoding
    df['Age'] = df['Age'].fillna(-1)
    df['Age'] = df['Age'].apply(lambda x: x / max(df['Age']))

    max_leucocytes = np.percentile(df['Leucocytes'].dropna(), 95)  # remove outliers
    max_lacticacid = np.percentile(df['LacticAcid'].dropna(), 95)  # remove outliers
    max_crp = np.percentile(df['CRP'].dropna(), 95)  # remove outliers

    x_seqs = []
    x_statics = []
    x_time_vals = []
    y = []

    for case in df['Case ID'].unique():

        after_registration_flag = False
        found_target_flag = False

        df_tmp = df[df['Case ID'] == case]
        df_tmp = df_tmp.sort_values(by='Complete Timestamp')
        idx = -1
        current_leucocytes_value = 0
        current_crp_value = 0
        current_lacticacid_value = 0

        for _, x in df_tmp.iterrows():
            idx = idx + 1
            if x['Activity'] == 'ER Registration' and idx == 0:
                x_statics.append(x[static_features].values.astype(float))
                x_time_vals.append([])
                x_seqs.append([])
                after_registration_flag = True

            if x['Activity'] == target_activity and after_registration_flag:
                found_target_flag = True

            if after_registration_flag:
                one_hot, current_leucocytes_value, current_crp_value, current_lacticacid_value = \
                    util.get_one_hot_of_activity_sepsis(x, max_leucocytes, max_crp, max_lacticacid,
                                                        current_leucocytes_value, current_crp_value,
                                                        current_lacticacid_value)
                x_seqs[-1].append(one_hot)
                x_time_vals[-1].append(x['Complete Timestamp'])

        if after_registration_flag:
            if found_target_flag:
                y.append(1)
            else:
                y.append(0)

    assert len(x_seqs) == len(x_statics) == len(y) == len(x_time_vals)

    x_seqs_, x_statics_, y_, x_time_vals_ = [], [], [], []
    for i, x in enumerate(x_seqs):
        if min_len <= len(x) <= max_len:
            x_seqs_.append(x)
            x_statics_.append(x_statics[i])
            y_.append(y[i])
            x_time_vals_.append(x_time_vals[i])

    """
    # Create event log
    f = open(f'../output/sepsis.txt', "w+")
    f.write(f'Case ID, Activity, Timestamp,{",".join([x for x in static_features])} \n')
    for idx in range(0, len(x_seqs_)):
        for idx_ts in range(0, len(x_seqs_[idx])):
            f.write(f'{idx},{int2act[np.argmax(x_seqs_[idx][idx_ts])]},'
                    f'{x_time_vals_[idx][idx_ts]},{",".join([str(x) for x in x_statics_[idx]])}\n')
    f.close()
    """

    return x_seqs_, x_statics_, y_, x_time_vals_, seq_features, static_features


def get_mimic_data(target_activity, max_len, min_len):
    ds_path = '../data/mimic_admission_activities_cleaned_short_final.csv'

    static_features = ['gender', 'insurance']

    static_bin_features = ['diagnosis_GASTROINTESTINAL BLEED', 'diagnosis_FEVER', 'diagnosis_ABDOMINAL PAIN']

    seq_act_features = ['PHYS REFERRAL/NORMAL DELI', 'HOME', 'EMERGENCY ROOM ADMIT', 'SNF',
                        'HOME WITH HOME IV PROVIDR', 'HOME HEALTH CARE', 'DEAD/EXPIRED',
                        'SHORT TERM HOSPITAL', 'TRANSFER FROM HOSP/EXTRAM', 'REHAB/DISTINCT PART HOSP',
                        'DISC-TRAN CANCER/CHLDRN H', 'CLINIC REFERRAL/PREMATURE', 'LONG TERM CARE HOSPITAL',
                        'DISC-TRAN TO FEDERAL HC', 'HOSPICE-MEDICAL FACILITY', 'LEFT AGAINST MEDICAL ADVI',
                        'HOSPICE-HOME', 'TRANSFER FROM OTHER HEALT', 'DISCH-TRAN TO PSYCH HOSP',
                        'TRANSFER FROM SKILLED NUR', 'HMO REFERRAL/SICK', '** INFO NOT AVAILABLE **',
                        'OTHER FACILITY', 'ICF', 'SNF-MEDICAID ONLY CERTIF', 'TRSF WITHIN THIS FACILITY']

    seq_features = []

    int2act = dict(zip(range(len(seq_act_features)), seq_act_features))

    static_features = static_bin_features + static_features
    seq_features = seq_act_features + seq_features

    df = pd.read_csv(ds_path)

    idx = 0
    for case in df['Case ID'].unique():
        df_tmp = df[df['Case ID'] == case]
        df_tmp = df_tmp.sort_values(by='Complete Timestamp')

        # Gender
        if len(df_tmp['gender'].unique()) > 1:
            values = df_tmp['gender'].unique().tolist()
            values = [x for x in values if str(x) != 'nan']
            try:
                if len(values) > 0:
                    df_tmp.loc[:, 'gender'] = values[0]
            except:
                pass

        if idx == 0:
            df_tmp_complete = df_tmp
        else:
            df_tmp_complete = pd.concat([df_tmp_complete, df_tmp])
        idx = idx + 1
    df = df_tmp_complete

    # Remove irrelevant data
    remove_cols = ['dob', 'dod', 'dod_hosp', 'age_dead', 'admission_type', 'ethnicity',
                   'marital_status', 'language', 'religion', 'age',
                   'diagnosis_NEWBORN', 'diagnosis_PNEUMONIA', 'diagnosis_SEPSIS',
                           'diagnosis_CORONARY ARTERY DISEASE', 'diagnosis_CONGESTIVE HEART FAILURE',
                           'diagnosis_CHEST PAIN',
                           'diagnosis_INTRACRANIAL HEMORRHAGE', 'diagnosis_ALTERED MENTAL STATUS',
                           'diagnosis_UPPER GI BLEED',
                           'diagnosis_CORONARY ARTERY DISEASECORONARY ARTERY BYPASS GRAFT /SDA',
                           'diagnosis_STROKE', 'diagnosis_HYPOTENSION']
    remove_cols = remove_cols
    df = df.drop(columns=remove_cols)

    # Time feature
    df['Complete Timestamp'] = pd.to_datetime(df['Complete Timestamp'])

    cat_features = ['gender', 'insurance']

    # Cat features
    for cat_feature in cat_features:
        mapping = dict(zip(df[cat_feature].unique(), np.arange(len(df[cat_feature].unique()))))  # ordinal encoding
        df[cat_feature] = df[cat_feature].apply(lambda x: mapping[x])
        max_ = max(df[cat_feature])
        df[cat_feature] = df[cat_feature].apply(lambda x: x / max_)  # normalise ordinal encoding

    # Bin features
    for bin_feature in static_bin_features:
        df[bin_feature] = df[bin_feature].fillna(0)

    x_seqs = []
    x_statics = []
    x_time_vals = []
    y = []
    x_statics_val_corr = []

    for case in df['Case ID'].unique():

        after_registration_flag = False
        found_target_flag = False

        df_tmp = df[df['Case ID'] == case]
        df_tmp = df_tmp.sort_values(by='Complete Timestamp')

        idx = -1
        x_past = 0
        for _, x in df_tmp.iterrows():
            idx = idx + 1
            if idx == 0:
                x_statics.append(x[static_features].values.astype(float))
                x_time_vals.append([])
                x_statics_val_corr.append([])
                x_seqs.append([])
                after_registration_flag = True
                x_past = x

            if x['Activity'] == target_activity and after_registration_flag:
                found_target_flag = True

            if after_registration_flag:

                x_seqs[-1].append(np.array(list(util.get_one_hot_of_activity_mimic(x))))
                x_time_vals[-1].append(x['Complete Timestamp'])

                for static_feature_ in static_features:
                    if static_feature_ in static_bin_features:
                        if x_past[static_feature_] == 1:
                            x[static_feature_] = 1
                    else:
                        pass

                x_statics_val_corr[-1].append(x[static_features])
                x_past = x

        if after_registration_flag:
            if found_target_flag:
                y.append(1)
            else:
                y.append(0)

    assert len(x_seqs) == len(x_statics) == len(y) == len(x_time_vals) == len(x_statics_val_corr)

    x_seqs_, x_statics_, y_, x_time_vals_, x_statics_val_corr_ = [], [], [], [], []
    for i, x in enumerate(x_seqs):
        if min_len <= len(x) <= max_len:
            x_seqs_.append(x)
            x_statics_.append(x_statics[i])
            y_.append(y[i])
            x_time_vals_.append(x_time_vals[i])
            x_statics_val_corr_.append(x_statics_val_corr[i])

    return x_seqs_, x_statics_, y_, x_time_vals_, seq_features, static_features, x_statics_val_corr_
