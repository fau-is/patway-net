import pandas as pd
import src.util as util
import numpy as np
from sklearn import ensemble


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
                       'Oligurie', 'DiagnosticLacticAcid', 'Hypoxie', #'Diagnose'
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

    """
    def map_diagnose_to_bin_features(df, feature, static_features):

        min_vals = 3
        most_important_features = 5
        df_one_hot_all = pd.get_dummies(df[feature], prefix=feature, dummy_na=False)
        df_one_hot = df_one_hot_all

        # get bin features with min_vals
        for col in df_one_hot.columns:
            if df_one_hot[col].value_counts().tolist()[1] < min_vals:
                df_one_hot.drop(col, inplace=True, axis=1)

        col_values = [v.replace("Diagnose_", "") for v in df_one_hot.columns.tolist()]

        # get label and data
        x_statics = []
        y = []
        for case in df['Case ID'].unique():
            df_tmp = df[df['Case ID'] == case]
            df_tmp = df_tmp.sort_values(by='Complete Timestamp')
            idx = -1
            for _, x in df_tmp.iterrows():
                idx = idx + 1
                if x['Activity'] == 'ER Registration' and idx == 0 and min_len <= len(df_tmp) <= max_len:

                    if x[feature] in col_values:
                        x_statics.append(x[feature])
                        if target_activity in df_tmp["Activity"].unique():
                            y.append(1)
                        else:
                            y.append(0)

        df_data = pd.DataFrame(x_statics, columns=["Diagnose"])
        df_data = pd.get_dummies(df_data, dummy_na=False)
        X = df_data.to_numpy()

        clf = ensemble.RandomForestClassifier()
        clf.fit(X, y)
        fi = clf.feature_importances_

        fi_df = pd.DataFrame(fi, columns=['Importances'])
        fi_df["Feature_Names"] = df_one_hot.columns
        fi_df = fi_df.nlargest(n=most_important_features, columns=['Importances'])
        fi = fi_df["Feature_Names"].tolist()

        # update df
        df.drop(feature, inplace=True, axis=1)
        df = pd.concat([df, df_one_hot_all[fi]], axis=1)

        # update static features
        static_features.remove("Diagnose")
        static_features = static_features + fi

        return df, static_features

    df, static_features = map_diagnose_to_bin_features(df, "Diagnose", static_features)
    """

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
                if not found_target_flag:
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
