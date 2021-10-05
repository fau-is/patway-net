import pandas as pd
from sklearn.preprocessing import RobustScaler
import src.util as util
import numpy as np


def get_sepsis_data(target_activity, max_len, min_len):
    ds_path = '../data/Sepsis Cases - Event Log.csv'

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

    # pre-processing
    df = pd.read_csv(ds_path)

    # sort case id by timestamp of first event
    df_ = df.groupby('Case ID').first()
    df_ = df_.sort_values(by='Complete Timestamp')
    x = pd.CategoricalDtype(df_.index.values, ordered=True)
    df['Case ID'] = df['Case ID'].astype(x)
    df = df.sort_values(['Case ID', 'Complete Timestamp'])
    df = df.reset_index()

    df['Complete Timestamp'] = pd.to_datetime(df['Complete Timestamp'])
    diagnose_mapping = dict(zip(df['Diagnose'].unique(), np.arange(len(df['Diagnose'].unique()))))  # ordinal encoding
    df['Diagnose'] = df['Diagnose'].apply(lambda x: diagnose_mapping[x])
    df['Diagnose'] = df['Diagnose'].apply(lambda x: x / max(df['Diagnose']))  # normalise ordinal encoding
    df['Age'] = df['Age'].fillna(-1)
    df['Age'] = df['Age'].apply(lambda x: x / max(df['Age']))

    max_leucocytes = np.percentile(df['Leucocytes'].dropna(), 95)  # remove outliers
    max_lacticacid = np.percentile(df['LacticAcid'].dropna(), 95)  # remove outliers

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
                x_seqs[-1].append(util.get_custom_one_hot_of_activity(x, max_leucocytes, max_lacticacid))
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

    # create event log
    f = open(f'../output/sepsis.txt', "w+")
    f.write(f'Case ID, Activity, Timestamp,{",".join([x for x in static_features])} \n')
    for idx in range(0, len(x_seqs_)):
        for idx_ts in range(0, len(x_seqs_[idx])):
            f.write(f'{idx},{int2act[np.argmax(x_seqs_[idx][idx_ts])]},'
                    f'{x_time_vals_[idx][idx_ts]},{",".join([str(x) for x in x_statics_[idx]])}\n')
    f.close()

    return x_seqs_, x_statics_, y_, x_time_vals_, seq_features, static_features


def get_mimic_data(target, max_len, min_len):

    ds_path = '../data/mimic_admission_activities_cleaned_final.csv'


    static_bin_features = ['diagnosis_NEWBORN', 'diagnosis_PNEUMONIA', 'diagnosis_SEPSIS',
                       'diagnosis_CORONARY ARTERY DISEASE', 'diagnosis_CONGESTIVE HEART FAILURE',
                       'diagnosis_CHEST PAIN', 'diagnosis_GASTROINTESTINAL BLEED',
                       'diagnosis_INTRACRANIAL HEMORRHAGE', 'diagnosis_ALTERED MENTAL STATUS',
                       'diagnosis_FEVER', 'diagnosis_ABDOMINAL PAIN',
                       'diagnosis_UPPER GI BLEED',
                       'diagnosis_CORONARY ARTERY DISEASE\CORONARY ARTERY BYPASS GRAFT /SDA',
                       'diagnosis_STROKE', 'diagnosis_HYPOTENSION']

    static_features = ['ethnicity', 'gender', 'language']

    seq_act_features = ['PHYS REFERRAL/NORMAL DELI', 'HOME', 'EMERGENCY ROOM ADMIT', 'SNF',
                        'HOME WITH HOME IV PROVIDR', 'HOME HEALTH CARE', 'DEAD/EXPIRED',
                        'SHORT TERM HOSPITAL', 'TRANSFER FROM HOSP/EXTRAM', 'REHAB/DISTINCT PART HOSP',
                        'DISC-TRAN CANCER/CHLDRN H', 'CLINIC REFERRAL/PREMATURE', 'LONG TERM CARE HOSPITAL',
                        'DISC-TRAN TO FEDERAL HC', 'HOSPICE-MEDICAL FACILITY', 'LEFT AGAINST MEDICAL ADVI',
                        'HOSPICE-HOME', 'TRANSFER FROM OTHER HEALT', 'DISCH-TRAN TO PSYCH HOSP',
                        'TRANSFER FROM SKILLED NUR', 'HMO REFERRAL/SICK', '** INFO NOT AVAILABLE **',
                        'OTHER FACILITY', 'ICF', 'SNF-MEDICAID ONLY CERTIF', 'TRSF WITHIN THIS FACILITY']

    seq_features = ['admission_type', 'insurance', 'marital_status', 'religion', 'age', 'age_dead']

    int2act = dict(zip(range(len(seq_act_features)), seq_act_features))

    static_features = static_features + static_bin_features
    seq_features = seq_features + seq_act_features

    # pre-processing
    df = pd.read_csv(ds_path)

    # ids = df['id'].unique()
    # ids_dic = dict(zip(df['id'].unique(), [0] * len(df['id'].unique())))

    # sort case id by timestamp of first event
    df = df.sort_values(['id', 'time'])
    df = df.reset_index()

    # remove irrelevant data
    df = df.drop(columns=['dob', 'dod', 'dod_hosp', 'index'])

    # time feature
    df['time'] = pd.to_datetime(df['time'])
    cat_features = ['admission_type', 'insurance', 'language', 'religion', 'marital_status', 'religion', 'ethnicity', 'gender']

    # cat features
    for cat_feature in cat_features:
        mapping = dict(zip(df[cat_feature].unique(), np.arange(len(df[cat_feature].unique()))))  # ordinal encoding
        df[cat_feature] = df[cat_feature].apply(lambda x: mapping[x])
        # df[cat_feature] = df[cat_feature].apply(lambda x: x / max(df[cat_feature]))  # normalise ordinal encoding

    # num features
    df['age'] = df['age'].fillna(-1)
    df['age_dead'] = df['age_dead'].fillna(-1)
    df['age'] = df['age'].apply(lambda x: x / max(df['age']))
    df['age_dead'] = df['age_dead'].apply(lambda x: x / max(df['age_dead']))

    # bin features
    bin_features = static_bin_features

    print(0)


    # return x_seqs_, x_stats_, y_, seq_features, static_features

    return 0