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

    seq_features = ['Leucocytes', 'CRP', 'LacticAcid', 'ER Triage', 'ER Sepsis Triage',
                    'IV Liquid', 'IV Antibiotics', 'Admission NC', 'Admission IC',
                    'Return ER', 'Release A', 'Release B', 'Release C', 'Release D',
                    'Release E']

    # pre-processing
    df = pd.read_csv(ds_path)
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
        for _, x in df_tmp.iterrows():
            if x['Activity'] == 'ER Registration':
                x_statics.append(x[static_features].values.astype(float))
                x_time_vals.append([])
                x_seqs.append([])
                after_registration_flag = True
                continue

            if 'Release' in x['Activity']:
                if x['Activity'] == target_activity:
                    y.append(1)
                    found_target_flag = True
                break

            if x['Activity'] == target_activity:
                y.append(1)
                found_target_flag = True
                break

            if after_registration_flag:
                x_seqs[-1].append(util.get_custom_one_hot_of_activity(x, max_leucocytes, max_lacticacid))
                x_time_vals[-1].append(x['Complete Timestamp'])

        if not found_target_flag and after_registration_flag:
            y.append(0)

    assert len(x_seqs) == len(x_statics) == len(y) == len(x_time_vals)

    print(f'Cutting everything after {max_len} activities')
    x_seqs_, x_statics_, y_, x_time_vals_ = [], [], [], []
    for i, x in enumerate(x_seqs):
        if min_len <= len(x) <= max_len:
            x_seqs_.append(x)
            x_statics_.append(x_statics[i])
            y_.append(y[i])
            x_time_vals_.append(x_time_vals[i])

    x_seqs_final = np.zeros((len(x_seqs_), max_len, len(x_seqs_[0][0])), dtype=np.float32)
    for i, x in enumerate(x_seqs):
        x_seqs_final[i, :len(x), :] = np.array(x)

    x_statics_final = np.array(x_statics_)
    y_final = np.array(y_).astype(np.int32)

    return x_seqs_final, x_statics_final, y_final, x_time_vals_, seq_features, static_features


def get_data_mimic(target, max_len, min_len):

    def cut_at_target(l):
        try:
            target_index = l.index(target) + 1
        except:
            target_index = None
        return l[:target_index]

    static_features = ['gender', 'anchor_age']

    seq_features = ['Neurology', 'Vascular', 'Medicine', 'PACU', 'Cardiac Surgery', 'Thoracic Surgery', 'Labor & Delivery',
                    'Surgery/Trauma', 'Trauma SICU TSICU', 'Med/Surg', 'Hematology/Oncology', 'Transplant', 'Nursery , Well Babies',
                    'Surgery', 'Med/Surg/Trauma', 'Psychiatry', 'Med/Surg/GYN', 'Observation', 'Surgical ensive Care Unit SICU',
                    'Medical/Surgical Gynecology', 'Medical ensive Care Unit MICU', 'Medicine/Cardiology', 'Coronary Care Unit CCU',
                    'Surgery/Pancreatic/Biliary/Bariatric', 'Medical/Surgical ensive Care Unit MICU/SICU', 'Neonatal ensive Care Unit NICU',
                    'Emergency Department Observation', 'Cardiac Vascular ensive Care Unit CVICU', 'Obstetrics Postpartum & Antepartum',
                    'Unknown', 'Special Care Nursery SCN', 'Neuro Surgical ensive Care Unit Neuro SICU', 'Neuro Stepdown',
                    'Obstetrics Antepartum', 'Cardiology', 'Obstetrics Postpartum', 'Medicine/Cardiology ermediate',
                    'Neuro ermediate', 'Hematology/Oncology ermediate', 'Surgery/Vascular/ermediate',
                    'Cardiology Surgery ermediate']

    df = pd.read_csv('../data/transfers.csv.gz')

    df = df[~pd.isnull(df.careunit)]
    df = df[df.eventtype == 'admit']

    carunit2ind = dict(zip(df.careunit.unique(), range(len(df.careunit.unique()))))
    max_index = len(df.careunit.unique()) + 1

    subj_id_to_seq = df.groupby(by='subject_id').agg({'careunit': list})

    subj_id_to_seq['careunit'] = subj_id_to_seq.careunit.apply(cut_at_target)
    subj_id_to_seq['len'] = subj_id_to_seq.careunit.apply(lambda x: len(x))
    subj_id_to_seq = subj_id_to_seq[subj_id_to_seq.len >= min_len]
    subj_id_to_seq = subj_id_to_seq[subj_id_to_seq.len <= max_len]

    subj_id_to_seq['y'] = subj_id_to_seq.careunit.apply(lambda x: 1 if x[-1] == target else 0)
    subj_id_to_seq['careunit'] = subj_id_to_seq.careunit.apply(lambda x: x[:-1])

    df = pd.read_csv('../data/patients.csv.gz')

    subj_id_to_stat = df.groupby(by='subject_id').agg({'gender': 'first', 'anchor_age': 'mean'})

    Xy = subj_id_to_seq.join(subj_id_to_stat)

    x_seqs = pd.DataFrame(Xy.careunit.tolist())
    x_seqs = x_seqs.fillna(max_index - 1)
    for c in x_seqs.columns:
        x_seqs[c] = x_seqs[c].replace(carunit2ind)

    x_seqs_ext = np.zeros((len(x_seqs), max_len, max_index), dtype=np.float32)

    for c in x_seqs.columns:
        x_seqs_ext[:, c, :] = np.eye(max_index)[x_seqs[c]]

    x_seqs_ext = x_seqs_ext[:, :, :-1]

    Xy['gender'] = Xy['gender'].apply(lambda x: 1 if x == 'F' else 0)
    Xy['anchor_age'] = RobustScaler().fit_transform(Xy['anchor_age'].values.reshape(-1, 1))
    x_static = Xy[['gender', 'anchor_age']].values

    y = Xy['y'].values.reshape(-1, 1)

    return x_seqs_ext, x_static, y, seq_features, static_features