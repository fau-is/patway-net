import pandas as pd
import src.util as util
import numpy as np
from sklearn import ensemble


def get_sim_data(label, file):
    ds_path = f'../data/{file}'

    static_features = ['Gender', 'Foreigner', 'BMI', 'Age']
    seq_features = ['ER Registration', 'Medication B', 'Medication A', 'Heart Rate', 'Blood Pressure']

    df = pd.read_csv(ds_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    age_max = max(df['Age'])
    df['Age'] = df['Age'].apply(lambda x: x / age_max)
    bmi_max = max(df['BMI'])
    df['BMI'] = df['BMI'].apply(lambda x: x / bmi_max)

    max_blood_pressure = np.percentile(df['Blood Pressure'].dropna(), 100)  # remove outliers
    max_heart_rate = np.percentile(df['Heart Rate'].dropna(), 100)  # remove outliers

    x_seqs = []
    x_statics = []
    x_time_vals = []
    y = []

    for case in df['Case ID'].unique():

        after_registration_flag = False
        df_tmp = df[df['Case ID'] == case]
        idx = -1
        current_heart_rate_value = 0
        current_blood_pressure_value = 0

        for _, x in df_tmp.iterrows():
            idx = idx + 1
            if x['Activity'] == 'ER Registration' and idx == 0:
                x_statics.append(x[static_features].values.astype(float))
                x_time_vals.append([])
                x_seqs.append([])
                after_registration_flag = True

            if after_registration_flag:
                one_hot, current_heart_rate_value, current_blood_pressure_value = util.get_one_hot_of_activity_sim(x, max_blood_pressure, max_heart_rate, current_blood_pressure_value, current_heart_rate_value)
                x_seqs[-1].append(one_hot)
                x_time_vals[-1].append(x['Timestamp'])

        if after_registration_flag:
            y.append(x[label])


    assert len(x_seqs) == len(x_statics) == len(y) == len(x_time_vals)

    """
    # Create event log
    f = open(f'../output/sim.txt', "w+")
    f.write(
        f'Case ID, Activity, Timestamp,{",".join([x for x in static_features])},{",".join([x for x in seq_features])},Label \n')
    for idx in range(0, len(x_seqs)):
        for idx_ts in range(0, len(x_seqs[idx])):
            f.write(f'{idx},'
                    f'{acts[idx][idx_ts]},'
                    f'{x_time_vals[idx][idx_ts]},'
                    f'{",".join([str(x) for x in x_statics[idx]])},'
                    f'{",".join([str(x) for x in x_seqs[idx][idx_ts]])},'
                    f'{y[idx]}\n')
    f.close()
    """

    return x_seqs, x_statics, y, x_time_vals, seq_features, static_features


def get_sepsis_data(target_activity, max_len, min_len):
    ds_path = '../../data/Sepsis Cases - Event Log_end2.csv'

    static_features = ['InfectionSuspected', 'DiagnosticBlood', 'DisfuncOrg',
                       'SIRSCritTachypnea', 'Hypotensie',
                       'SIRSCritHeartRate', 'Infusion', 'DiagnosticArtAstrup', 'Age',
                       'DiagnosticIC', 'DiagnosticSputum', 'DiagnosticLiquor',
                       'DiagnosticOther', 'SIRSCriteria2OrMore', 'DiagnosticXthorax',
                       'SIRSCritTemperature', 'DiagnosticUrinaryCulture', 'SIRSCritLeucos',
                       'Oligurie', 'DiagnosticLacticAcid', 'Hypoxie',  # 'Diagnose',
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
    max_age = max(df['Age'])
    df['Age'] = df['Age'].apply(lambda x: x / max_age)

    max_leucocytes = np.percentile(df['Leucocytes'].dropna(), 95)  # remove outliers
    max_lacticacid = np.percentile(df['LacticAcid'].dropna(), 95)  # remove outliers
    max_crp = np.percentile(df['CRP'].dropna(), 95)  # remove outliers

    """
    mean_leucocytes = sum(df["Leucocytes"].loc[df['Leucocytes'] <= max_leucocytes]) / len(df["Leucocytes"].loc[df['Leucocytes'] <= max_leucocytes])
    mean_lacticacid = sum(df["LacticAcid"].loc[df['LacticAcid'] <= max_lacticacid]) / len(
        df["LacticAcid"].loc[df['LacticAcid'] <= max_lacticacid])
    mean_crp = sum(df["CRP"].loc[df['CRP'] <= max_crp]) / len(df["CRP"].loc[df['CRP'] <= max_crp])
    """

    x_seqs = []
    x_statics = []
    x_time_vals = []
    y = []
    acts = []

    for case in df['Case ID'].unique():

        after_registration_flag = False
        found_target_flag = False

        df_tmp = df[df['Case ID'] == case]
        df_tmp = df_tmp.sort_values(by='Complete Timestamp')
        idx = -1
        current_leucocytes_value = 0
        current_crp_value = 0
        current_lacticacid_value = 0

        """
        # backward imputation same value
        df_tmp["Leucocytes"] = (df_tmp["Leucocytes"].replace(to_replace=np.nan, method='ffill')).replace(to_replace=np.nan, method='bfill')
        df_tmp["LacticAcid"] = (df_tmp["LacticAcid"].replace(to_replace=np.nan, method='ffill')).replace(to_replace=np.nan, method='bfill')
        df_tmp["CRP"] = (df_tmp["CRP"].replace(to_replace=np.nan, method='ffill')).replace(to_replace=np.nan, method='bfill')
        """

        """
        # backward imputation mean
        df_tmp["Leucocytes"] = (df_tmp["Leucocytes"].replace(to_replace=np.nan, method='ffill')).fillna(mean_leucocytes)
        df_tmp["LacticAcid"] = (df_tmp["LacticAcid"].replace(to_replace=np.nan, method='ffill')).fillna(mean_lacticacid)
        df_tmp["CRP"] = (df_tmp["CRP"].replace(to_replace=np.nan, method='ffill')).fillna(mean_crp)
        """

        for _, x in df_tmp.iterrows():
            idx = idx + 1
            if x['Activity'] == 'ER Registration' and idx == 0:
                x_statics.append(x[static_features].values.astype(float))
                x_time_vals.append([])
                x_seqs.append([])
                after_registration_flag = True
                acts.append([])

            if x['Activity'] == target_activity and after_registration_flag:
                found_target_flag = True

            if after_registration_flag:
                if not found_target_flag:
                    one_hot, current_leucocytes_value, current_crp_value, current_lacticacid_value = (
                        util.get_one_hot_of_activity_sepsis(x, max_leucocytes, max_crp, max_lacticacid,
                                                            current_leucocytes_value, current_crp_value,
                                                            current_lacticacid_value))
                    x_seqs[-1].append(one_hot)
                    x_time_vals[-1].append(x['Complete Timestamp'])
                    acts[-1].append(x['Activity'])

        if after_registration_flag:
            if found_target_flag:
                y.append(1)
            else:
                y.append(0)

    assert len(x_seqs) == len(x_statics) == len(y) == len(x_time_vals)

    x_seqs_, x_statics_, y_, x_time_vals_, acts_ = [], [], [], [], []
    for i, x in enumerate(x_seqs):
        if min_len <= len(x) <= max_len:
            x_seqs_.append(x)
            x_statics_.append(x_statics[i])
            y_.append(y[i])
            x_time_vals_.append(x_time_vals[i])
            acts_.append(acts[i])

    """
    # Create event log
    f = open(f'../output/sepsis.txt', "w+")
    f.write(f'Case ID, Activity, Timestamp,{",".join([x for x in static_features])},'
            f'{",".join([x for x in seq_features])},Label \n')
    for idx in range(0, len(x_seqs_)):
        for idx_ts in range(0, len(x_seqs_[idx])):
            f.write(f'{idx},{acts_[idx][idx_ts]},'
                    f'{x_time_vals_[idx][idx_ts]},'
                    f'{",".join([str(x) for x in x_statics_[idx]])},'
                    f'{",".join([str(x) for x in x_seqs_[idx][idx_ts]])},'
                    f'{y_[idx]}\n')
    f.close()
    """

    """
    from apyori import apriori
    association_rules = apriori(acts_, min_support=0.01, min_confidence=0.01, min_lift=1, min_length=2)
    association_results = list(association_rules)

    for item in association_rules:
        # first index of the inner list
        # Contains base item and add item
        pair = item[0]
        items = [x for x in pair]
        print("Rule: " + items[0] + " -> " + items[1])

        # second index of the inner list
        print("Support: " + str(item[1]))

        # third index of the list located at 0th
        # of the third index of the inner list

        print("Confidence: " + str(item[2][0][2]))
        print("Lift: " + str(item[2][0][3]))
        print("=====================================")
    """

    return x_seqs_, x_statics_, y_, x_time_vals_, seq_features, static_features


def get_bpi_data(max_len, min_len):
    # https://dl.acm.org/doi/abs/10.1145/3301300
    # https://github.com/irhete/predictive-monitoring-benchmark

    ds_path = '../data/bpic2012_O_ACCEPTED-COMPLETE.csv'
    static_features = ['AMOUNT_REQ']
    seq_features = ['Activity']

    df = pd.read_csv(ds_path, sep=";")
    df['Complete Timestamp'] = pd.to_datetime(df['Complete Timestamp'])
    df['Resource'] = df['Resource'].astype('category') # Resource is integer, but categorical according to TU Eindhoven and Teinema
    df['Activity'] = df['Activity'].astype('category')
    df['lifecycle:transition'] = df['lifecycle:transition'].astype('category')
    df['month'] = df['month'].astype('category')
    df['weekday'] = df['weekday'].astype('category')
    df['label'] = df['label'].replace({'deviant': 0, 'regular': 1}).astype('int64')

    df = df.sort_values(['Case ID', 'Complete Timestamp'])
    df = df.reset_index()

    # Remove resource as > 25 categories
    df = df.drop(['index', 'lifecycle:transition', 'timesincemidnight', 'timesincelastevent',
                  'timesincecasestart', 'event_nr', 'month', 'weekday', 'hour',
                  'open_cases', 'Resource'], axis=1)

    max_amt = max(df['AMOUNT_REQ'])
    df['AMOUNT_REQ'] = df['AMOUNT_REQ'].apply(lambda x: x / max_amt)

    df = pd.get_dummies(df)

    x_seqs = []
    x_statics = []
    x_time_vals = []
    y = []

    c = 0
    uni_cases = df['Case ID'].unique()
    for case in uni_cases:

        after_registration_flag = False
        df_tmp = df[df['Case ID'] == case]
        df_tmp = df_tmp.sort_values(by='Complete Timestamp')

        if min_len <= len(df_tmp) <= max_len:
            c += 1
        idx = 0
        print(f'{c} -- {len(uni_cases)}')

        for _, x in df_tmp.iterrows():
            if idx == 0:
                x_statics.append(x[static_features].values.astype(float))
                x_time_vals.append([])
                x_seqs.append([])
                after_registration_flag = True

            if after_registration_flag:
                    one_hot = x.drop(['Complete Timestamp', 'Case ID', 'label'] + static_features).values
                    x_seqs[-1].append(one_hot)
                    x_time_vals[-1].append(x['Complete Timestamp'])
                    label = x['label']
            idx = idx + 1

        if after_registration_flag:
            y.append(label)

        if c == 5000:
            break

    assert len(x_seqs) == len(x_statics) == len(y) == len(x_time_vals)

    x_seqs_, x_statics_, y_, x_time_vals_ = [], [], [], []
    for i, x in enumerate(x_seqs):
        if min_len <= len(x) <= max_len:
            x_seqs_.append(x)
            x_statics_.append(x_statics[i])
            y_.append(y[i])
            x_time_vals_.append(x_time_vals[i])

    return x_seqs_, x_statics_, y_, x_time_vals_, seq_features, static_features


def get_hospital_data(max_len, min_len):
    # https://dl.acm.org/doi/abs/10.1145/3301300
    # https://github.com/irhete/predictive-monitoring-benchmark

    ds_path = '../data/hospital_billing_2.csv'

    static_features = ['speciality', 'caseType']  # 'blocked', 'flagD'
    seq_features = ['Activity', 'state']  # 'actOrange', 'actRed', 'flagC', 'msgType', 'msgCode', 'version', 'isCancelled', 'msgCount'

    df = pd.read_csv(ds_path, sep=";")
    df['Complete Timestamp'] = pd.to_datetime(df['Complete Timestamp'])
    df['Activity'] = df['Activity'].astype('category')
    df['caseType'] = df['caseType'].astype('category')
    df['label'] = df['label'].replace({'deviant': 0, 'regular': 1}).astype('int64')

    df = df.sort_values(['Case ID', 'Complete Timestamp'])
    df = df.reset_index()

    # Remove resource, diagnosis, closeCode as > 25 categories
    df = df.drop(['index', 'timesincemidnight', 'timesincelastevent',
                  'timesincecasestart', 'event_nr', 'month', 'weekday', 'hour', 'open_cases',
                  'Resource', 'diagnosis', 'closeCode', 'actOrange', 'actRed', 'flagC',
                  'msgType', 'state', 'msgCode', 'version', 'isCancelled',
                  'msgCount', 'blocked', 'flagD'], axis=1)

    # max_amt = max(df['amount'])
    # df['amount'] = df['amount'].apply(lambda x: x / max_amt)

    not_transform_cols = ['Case ID', 'Complete Timestamp', 'label']
    df = pd.get_dummies(data=df, columns=[col for col in df.columns if col not in not_transform_cols])

    static_features_ = []
    for static_feature in static_features:
        static_features_ = static_features_ + [f for f in list(df.columns) if static_feature in f]
    static_features = static_features_

    x_seqs = []
    x_statics = []
    x_time_vals = []
    y = []

    c = 0
    uni_cases = df['Case ID'].unique()
    for case in uni_cases:

        after_registration_flag = False
        df_tmp = df.loc[df['Case ID'].isin([case])]
        df_tmp = df_tmp.sort_values(by='Complete Timestamp')

        if min_len <= len(df_tmp) <= max_len:
            c += 1

        idx = 0
        print(f'{c} -- {len(uni_cases)}')

        for _, x in df_tmp.iterrows():

            if idx == 0:
                x_statics.append(x[static_features].values.astype(float))
                x_time_vals.append([])
                x_seqs.append([])
                after_registration_flag = True

            if after_registration_flag:
                one_hot = x.drop(['Complete Timestamp', 'Case ID', 'label'] + static_features).values
                x_seqs[-1].append(one_hot)
                x_time_vals[-1].append(x['Complete Timestamp'])
                label = x['label']
            idx = idx + 1

        if after_registration_flag:
            y.append(label)

        if c == 5000:
            break

    assert len(x_seqs) == len(x_statics) == len(y) == len(x_time_vals)

    x_seqs_, x_statics_, y_, x_time_vals_ = [], [], [], []
    for i, x in enumerate(x_seqs):
        if min_len <= len(x) <= max_len:
            x_seqs_.append(x)
            x_statics_.append(x_statics[i])
            y_.append(y[i])
            x_time_vals_.append(x_time_vals[i])

    return x_seqs_, x_statics_, y_, x_time_vals_, seq_features, static_features