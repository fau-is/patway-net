import pandas as pd
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
                x_seqs[-1].append(util.get_one_hot_of_activity_sepsis(x, max_leucocytes, max_lacticacid))
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
    # create event log
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

    static_features = ['gender', 'ethnicity', 'age', 'marital_status', 'language', 'religion', 'insurance']  # 'gender', 'ethnicity']

    static_bin_features = ['diagnosis_NEWBORN', 'diagnosis_PNEUMONIA', 'diagnosis_SEPSIS',
                           'diagnosis_CORONARY ARTERY DISEASE', 'diagnosis_CONGESTIVE HEART FAILURE',
                           'diagnosis_CHEST PAIN', 'diagnosis_GASTROINTESTINAL BLEED',
                           'diagnosis_INTRACRANIAL HEMORRHAGE', 'diagnosis_ALTERED MENTAL STATUS',
                           'diagnosis_FEVER', 'diagnosis_ABDOMINAL PAIN',
                           'diagnosis_UPPER GI BLEED',
                           'diagnosis_CORONARY ARTERY DISEASECORONARY ARTERY BYPASS GRAFT /SDA',
                           'diagnosis_STROKE', 'diagnosis_HYPOTENSION']

    seq_act_features = ['PHYS REFERRAL/NORMAL DELI', 'HOME', 'EMERGENCY ROOM ADMIT', 'SNF',
                        'HOME WITH HOME IV PROVIDR', 'HOME HEALTH CARE', 'DEAD/EXPIRED',
                        'SHORT TERM HOSPITAL', 'TRANSFER FROM HOSP/EXTRAM', 'REHAB/DISTINCT PART HOSP',
                        'DISC-TRAN CANCER/CHLDRN H', 'CLINIC REFERRAL/PREMATURE', 'LONG TERM CARE HOSPITAL',
                        'DISC-TRAN TO FEDERAL HC', 'HOSPICE-MEDICAL FACILITY', 'LEFT AGAINST MEDICAL ADVI',
                        'HOSPICE-HOME', 'TRANSFER FROM OTHER HEALT', 'DISCH-TRAN TO PSYCH HOSP',
                        'TRANSFER FROM SKILLED NUR', 'HMO REFERRAL/SICK', '** INFO NOT AVAILABLE **',
                        'OTHER FACILITY', 'ICF', 'SNF-MEDICAID ONLY CERTIF', 'TRSF WITHIN THIS FACILITY']

    seq_features = []  # 'language', 'religion', 'marital_status', 'religion']

    int2act = dict(zip(range(len(seq_act_features)), seq_act_features))

    static_features = static_bin_features + static_features
    seq_features = seq_act_features + seq_features

    # pre-processing
    df = pd.read_csv(ds_path)

    """
    # todo check if seq features are really seq. features
    for feature in ['marital_status', 'language', 'religion']:
        sum_unique = 0
        sum_unique_wo_na = 0
        for case in df['Case ID'].unique():
            df_tmp = df[df['Case ID'] == case]
            df_tmp = df_tmp.sort_values(by='Complete Timestamp')

            tmp_unique = df_tmp[feature].unique()
            tmp_unique_wo_na = df_tmp[feature].dropna().unique()
            if len(tmp_unique) > 1:
                sum_unique = sum_unique + 1
            if len(tmp_unique_wo_na) > 1:
                sum_unique_wo_na = sum_unique_wo_na + 1
        print(0)
    """

    # todo: clean features
    idx = 0
    for case in df['Case ID'].unique():
        df_tmp = df[df['Case ID'] == case]
        df_tmp = df_tmp.sort_values(by='Complete Timestamp')

        # ethnicity
        if len(df_tmp['ethnicity'].unique()) > 1:
            values = df_tmp['ethnicity'].unique().tolist()
            if "UNKNOWN/NOT SPECIFIED" in values:
                values = df_tmp['ethnicity'].unique().tolist().remove("UNKNOWN/NOT SPECIFIED")
            try:
                if len(values) > 0:
                    df_tmp.loc[:, 'ethnicity'] = values[0]
            except:
                pass

        # gender
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

    # remove irrelevant data
    remove_cols = ['dob', 'dod', 'dod_hosp', 'age_dead', 'admission_type']
    remove_cols = remove_cols
    df = df.drop(columns=remove_cols)

    # time feature
    df['Complete Timestamp'] = pd.to_datetime(df['Complete Timestamp'])

    cat_features = ['ethnicity', 'gender', 'insurance', 'marital_status', 'language', 'religion']

    # cat features
    for cat_feature in cat_features:
        mapping = dict(zip(df[cat_feature].unique(), np.arange(len(df[cat_feature].unique()))))  # ordinal encoding
        df[cat_feature] = df[cat_feature].apply(lambda x: mapping[x])
        max_ = max(df[cat_feature])
        df[cat_feature] = df[cat_feature].apply(lambda x: x / max_)  # normalise ordinal encoding

    # num features
    df['age'] = df['age'].fillna(-1)
    _max = max(df['age'])
    df['age'] = df['age'].apply(lambda x: x / _max)

    # bin features
    for bin_feature in static_bin_features:
        df[bin_feature] = df[bin_feature].fillna(0)

    """
    # todo: create static features
    idx = 0
    for case in df['Case ID'].unique():
        df_tmp = df[df['Case ID'] == case]
        df_tmp = df_tmp.sort_values(by='Complete Timestamp')

        for static_bin_feature in static_bin_features:
            if sum(df_tmp[static_bin_feature][0:3]) > 0.0:
                df_tmp.loc[:, static_bin_feature] = 1.0

        if idx == 0:
            df_tmp_complete = df_tmp
        else:
            df_tmp_complete = pd.concat([df_tmp_complete, df_tmp])
        idx = idx + 1
    df = df_tmp_complete
    """

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

                """
                seq_features_vals = []
                for seq_feature_ in ['insurance']:
                    seq_features_vals.append(x[seq_feature_])
                """

                x_seqs[-1].append(np.array(list(util.get_one_hot_of_activity_mimic(x))))  # + seq_features_vals))
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


def get_bpi11_data(target_activity, max_len, min_len):
    ds_path = '../data/Hospital log_select_cols_seq_len_300.csv'

    static_features = ['Treatment code', 'Diagnosis code',
                       'Treatment code:3', 'Treatment code:2', 'Treatment code:1',
                       'Specialism code:3', 'Specialism code:2', 'Specialism code:1',
                       'Diagnosis code:3', 'Diagnosis code:2', 'Diagnosis code:1', 'Age',
                       'Diagnosis Treatment Combination ID', 'Diagnosis Treatment Combination ID:3',
                       'Diagnosis Treatment Combination ID:2', 'Diagnosis Treatment Combination ID:1',
                       'Diagnosis', 'Diagnosis:3', 'Diagnosis:2', 'Diagnosis:1',
                       'Diagnosis:4', 'Treatment code:4', 'Specialism code:4', 'Diagnosis code:4',
                       'Diagnosis Treatment Combination ID:4'
                       ]

    # seq_act_features = ['1e consult poliklinisch', 'administratief tarief       - eerste pol', 'verlosk.-gynaec. korte kaart kosten-out', 'echografie  - genitalia interna', 'simulator - gebruik voor aanvang megavol', 'behandeltijd - eenheid t3 - megavolt', 'teletherapie - megavolt fotonen bestrali', 'aanname laboratoriumonderzoek', 'ureum', 'hemoglobine foto-elektrisch', 'creatinine', 'natrium vlamfotometrisch', 'kalium potentiometrisch', 'leukocyten tellen elektronisch', 'trombocyten tellen - elektronisch', 'ordertarief', 'ligdagen - alle spec.beh.kinderg.-reval.', 'sediment - spoed', 'bacteriologisch onderzoek met kweek -nie', 'resistentiebepalingen - 5 bepalingen', 'hepatitis-b surface antigeen confirmatie', 'urine onderzoek - kwalitatief', '190021 klinische opname a002', '190205 klasse 3b        a205', 'sgot - asat kinetisch', 'sgpt - alat kinetisch', 'melkzuurdehydrogenase -ldh- kinetisch', 'differentiele telling automatisch', 'trombotest', 'crp c-reactief proteine', 'brachytherapie - interstitieel - intensi', 'inwend.geneesk.  korte kaart kosten-out', 'vervolgconsult poliklinisch', 'inwend.geneesk.   aanv.kaart kosten-out', 'squamous cell carcinoma mbv eia', 'ca-125 mbv meia', 'ct abdomen', 'telefonisch consult', 'patient niet verschenen radiologie', 'homocystine chromatografisch', 'vulva       - rad.vulvectomie - opp.en d', 'histologisch onderzoek - grote resectiep', 'sediment', 'microscopisch onderzoek - gekleurd en on', 'coupe ter inzage', 'bilirubine -geconjugeerd', 'bilirubine - totaal', 'glucose', 'alkalische fosfatase -kinetisch-', 'bloedgroep abo en rhesusfactor', 'rhesusfactor d - centrifugeermethode - e', 'gammaglutamyltranspeptidase', 'cea - tumormarker mbv meia', 'calcium', 'albumine', 'screening antistoffen erytrocyten', 'thorax', 'mri bekken', 'skelet - scintigrafie totale lichaam', 'verlosk.-gynaec.   jaarkaart kosten-out', 'cytologisch onderzoek - ectocervix -', '190101 bovenreg.toesl.  a101', 'vrw.gesl.org- biopsie-punctie-cytologie', 'vrw.gesl.org- curettage gefractioneerd', 'vulva       - biopsie-punctie-cytologie', 'uterus      - hysteroscopie', 'histologisch onderzoek - kleine resectie', 'ligase chain reaction -lcr-', 'dagverpleging - alle spec.beh.kind.-rev.', 'uterus      - intra-uterine device inbre', '190035 dagverpleging    a007', 'cytologisch onderzoek - vagina -', 'mammografie thoraxwand', 'bot - densitometrie lwk', 'bot - densitometrie femurhals', 'lymfeklier - scint sentinel node dynamis', 'lymfeklier - scint sentinel node vervolg', 'lymfeklier - scint sentinel node met pro', 'hematocriet mbv centrifuge', 'vulva       - radicale vulvect. znd ingu', 'vriescoupe', 'verlosk.-gynaec.  aanv.kaart kosten-out', 'fosfaat', 'alfa-1-foetoproteine', 'ct thorax', 'bezinkingssnelheid foto-elektrisch', 'bekken - staand', 'heup', 'mri wervelkolom - lumbaal', 'mri heup', 'magnesium  -aas', 'echo nieren-urinewegen', 'hematologie - ontstek.scint.tot.lichaam', 'haemoglobine foto-electrisch - spoed', 'ovarium     - adnex-extirpatie dmv lapar', '1e consult      bezoek', 'histologisch onderzoek - biopten nno', 'vulva       - vulvectomie zonder lieskli', 'immunopathologisch onderzoek', 'cito histologisch onderzoek', 'cytologisch onderzoek - diversen -', 'tsh enzym-immunologisch', 'vrij t4 mbv ria -ft 4', 'totaal t4 - thyroxine immunofluorimetris', 'e.c.g.      - elektrocardiografie', '190204 klasse 3a        a204', 'vagina      - biopsie-punctie-cytologie', 'blaas       - cystoscopie', 'vagina      - toucher onder anesthesie', 'blaas       - uretrocystoscopie nno', 'leucocyten tellen -electronisch- spoed', 'crp c-reactief proteine - spoed', 'pyelografie --wo niertransplantaat--', 'mri abdomen', 'cefalinetijd - coagulatie', 'protrombinetijd', 'colon inloop', 'ureum - spoed', 'creatinine - spoed', 'alkalische fosfatase - kinetisch - spoed', 'calcium - spoed', 'natrium - vlamfotometrisch - spoed', 'kalium vlamfotometrisch - spoed', 'totaal eiwit colorimetrisch - spoed', 'sgpt alat kinetisch - spoed', 'sgot asat kinetisch - spoed', 'trombocyten tellen - spoed', 'gamma glutamyltranspeptidase - spoed', 'albumine - spoed', 'magnesium  -aas - spoed', 'dieet nno', 'uterus      - extirpatie - abdominaal me', 'cytologisch onderzoek - ascites -', 'glucose - spoed', 'melkzuur enzymatisch - spoed', 'o2-saturatie', 'echo bovenbuik', 'behandeltijd - eenheid t2 - megavolt', 'triglyceriden - enzymatisch', 'vitamine b12 mbv chemieluminescentie', 'onderzoek van elders niet beoordeeld', 'bilirubine kwantitatief totaal of direct', 'urinezuur met uricase u.v. spectrofotome', 'melkzuur dehydrogenase - spoed', 'cholesterol totaal', 'betasubunit bep. mbv. ria hcg', 'cholesterol hdl', 'fsh enzym-immunologisch', 'lh mbv radioisotopen', 'progesteron mbv radioisotopen', 'prolactine mbv eia', 'oestradiol mbv radioisotopen -ria', 'analgesie   - epiduraal door anesthesist', 'vrw.gesl.org- plastische operatie van vu', 'bicarbonaat', 'actuele ph - pco2 - stand.bicarbonaat', 'testosteron mbv radioisotopen', 'sex-hormone-binding globuline mbv ria', 'urinewegen  - urodynamisch onderzoek-vij', 'differentiatie leukocyten - handmatig', 'punctie tbv cytologisch onderzoek door p', 'cytologisch onderzoek - urine niercyste', 'cervix      - lisexcisie portio - diathe', 'vagina      - scopie incl.evt.vulvabiops', 'protrombinetijd -quick owren of modif.-', 'heparinebepaling - kaoline - cefalinetij', 'international normalised ratio mbv tromb', 'vagina      - tension-free vaginal tape', 'vagina      - colpocleisis lefort', 'cde fenotypering', 'bloedgroepantigenen andere dan abo-rhesu', 'lymf.syst.  - stageringslymfadenectomie', 'widal agglutinatie - yersinia 3', 'widal agglutinatie - yersinia 9', 'amylase', 'foliumzuur mbv radioisotopen', 'dikke darm  - colonoscopie met biopsie', 'inwend.geneesk.    jaarkaart kosten-out', 'infuus      - inbrengen', 'hemoglobine a1', 'vulva       - vulvectomie - liesblok', 'ct bekken', 'echo mamma', 'interc.consult klinisch  anesthesie', 'wervelkolom - lumbaal', 'vulva       - ruime lokale excisie van a', 'behandeltijd - eenheid t1 - megavolt', 'mri grote hersenen', 'ct bovenbuik', 'kruisproef volledig -drie methoden-', '190034 afwezigheidsdag  a006', 'gefiltreerd erytrocytenconcentraat', 'interc.consult klinisch  chirurgie', 'alfa amylase - spoed', 'chloride - spoed', 'huidplastiek- z-plastiek huid hoofd en h', 'cytologisch onderzoek - lymfeklierpuncti', 'brain natriuretic peptide -bnp- mbv immu', 'vulva       - partiele vulvectomie', 'methemoglobine - sulfhemoglobine elk', 'co-hb kwn.', 'buikoverzicht', 'klinische kaart - anesthesie', 'uterus      - extirpatie abdominaal tota', 'cervix      - conisatie', 'eiwit -colorimetrisch- totaal', 'toonaudiometrie - audiologisch centrum', 'hoogfrequente audiometrie - audiologisch', 'patienten kontakt audiologie korter dan', 'anesthesie  - bij behandeling met radium', '17-hydroxyprogesteron mbv ria', 'delta-4-androsteendion', 'dhea-sulfaat met extractie mbv ria', 'ca-19.9 tumormarker', 'hematocriet - spoed', 'buik        - stageringslaparotom.-oment', 'vrw.gesl.org- exc.destruct.path.afwijkin', 'echografie  - blaas - extern', 'totaal eiwit', 'vitamine b1 - thiamine', 'bekken - liggend', 'anti-hav ig of igg-elisa of ria- in een', 'antist.tegen hepatitis-b surface antigee', 'hepatitis-b-anti-core igm', 'anti-hepatitis-c-virus mbv elisa', 'fosfaat - spoed', 'echogeleide als ass.bij punctie-biopsie', 'vagina      - operatie vesicovaginale fi', 'vagina      - scopie excl.weefselwegname', 'bovenarm', 'diagnostische punctie onder echocontrole', 'echo onderbuik', 'ct-gel. als ass. bij punctie - biopsie i', 'hyperthermie', 'cytologisch onderzoek - endocervix -', 'oestron mbv radioisotopen', 'vrw.gesl.org- adnex-extirpatie dmv lapar', 'cytologisch onderzoek - leverpunctie -', 'kreatinine fosfokinase - spoed', 'creatine fosfokinase mb - spoed', 'fibrinogeen', 'antitrombine iii mbv chromogeen-n-substr', 'vagina      - vaginotomie', 'ovarium     - redebulking ovarium carcin', 'directe antiglobulinetest - coombs', 'bloedgroepbep. anders dan abo per bloedg', 'antistoffen tegen erys - specificiteit -', 'uitsluiten irregulaire a.s. tegen erys d', 'ident. irregulaire a.s. tegen erys na po', 'bloedgroepbepaling anders dan abo  kidd', 'echografie a vue ivm zwangerschap - rout', 'echografie a vue ivw zwangerschap - met', 'duplexscan - venen been', 'vrw.gesl.org- adhesiolysis ovarium plus', 'chloride', 'klinische kaart - verloskunde en gynaeco', 'ct hals', 'ijzer', 'reticulocyten tellen mbv facscan', 'ferritine kwn mbv chemieluminescentie', 'transferrine mbv ics', 'behandeltijd - eenheid t4 - megavolt', 'mri wervelkolom - thoracaal', 'buik        - stag.lap-tumorred.-omentec', 'ct wervelkolom - lumbaal', 'ovarium     - debulking ovariumcarcinoom', 'echo abdomen', 'zwangerschapsreactie - spoed', 'ct pancreas en milt', 'ca 15.3 - tumormarker mbv meia', 'cytologisch onderzoek tbv bevolkingsonde', 'blaas       - cystoscopie met biopsie', 'onderz. naar aanwezigheid van koude anti', 'uterus      - extirpatie vaginaal totaal', 'nucleair onderzoek van elders', 'melkzuur enzymatisch', 'infuus      - bloed', 'echo schouder - m-z bovenarm', 'creatinefosfokinase -kinetisch-', 'creatine fosfokinase isoenzym ckmb', 'troponine-t mbv elisa', 'bovenbeen', 'peritoneum  - omentectomie', 'buik        - laparoscopie - diagnostisc', 'vrw.gesl.org- operatie van clitoris nno', 'cytologisch onderzoek - buiktumorpunctie', 'dikke darm  - appendectomie nno', 'buik        - adhesiolyse en biopsie dmv', 'schoudergewricht', 'endoretrograde cholangiografie', 'pyelografie antegraad via punctie', 'cystografie retrograad', 'echo hals --waaronder schildklier--', 'uterus      - carc. cervix vlgs. werthei', 'mri tractus urogenitalis', 'longfunctie - spirometrie', 'longfunctie - stroomvolume curve', 'interc.consult klinisch  radiotherapie', 'echo pancreas en milt', 'ro-gel. als ass. bij punctie - biopsie l', 'hemoglobinen -abnormaal incl. hba2', 'erytrocyten tellen - elektronisch', 'doorlichting zonder opname buik', 'ro-gel. als ass. bij inbrengen endoproth', 'lymf.syst.  - regionale lymfeklierdissec', 'echo buikwand', 'lymf.syst.  - sentinel node procedure li', 'ct hersenen', 'niet declarabele dagverpleging-bv klin.p', 'tumor - petscan totale lichaam', 'anti-hiv mbv elisa', 'bekken', 'rontgenonderzoek onvolledig', 'vrw.gesl.org- second-look-oper.ovariumca', 'echo been', 'anesthesie  - epiduraal met bloedpatch', 'paclitaxel', 'uterus      - tot. laparoscopische hyste', 'wervelkolom - cervicaal', 'wervelkolom - thoracaal', 'wervelkolom - thoracolumbale overgang', 'eerste hulp - niet seh afdeling - elders', 'itraconazol mbv hplc', 'bepaling hepatitis-be-antigeen', 'bepaling antist. hepatitis-be-antigeen']
    # seq_act_features = ['1e consult poliklinisch','administratief tarief       - eerste pol','verlosk.-gynaec. korte kaart kosten-out','echografie  - genitalia interna','simulator - gebruik voor aanvang megavol','behandeltijd - eenheid t3 - megavolt','teletherapie - megavolt fotonen bestrali','aanname laboratoriumonderzoek','ureum','hemoglobine foto-elektrisch','creatinine','natrium vlamfotometrisch','kalium potentiometrisch','leukocyten tellen elektronisch','trombocyten tellen - elektronisch','ordertarief','ligdagen - alle spec.beh.kinderg.-reval.','sediment - spoed','bacteriologisch onderzoek met kweek -nie','resistentiebepalingen - 5 bepalingen','hepatitis-b surface antigeen confirmatie','urine onderzoek - kwalitatief','190021 klinische opname a002','190205 klasse 3b        a205','sgot - asat kinetisch','sgpt - alat kinetisch','melkzuurdehydrogenase -ldh- kinetisch','differentiele telling automatisch','trombotest','crp c-reactief proteine','brachytherapie - interstitieel - intensi','inwend.geneesk.  korte kaart kosten-out','vervolgconsult poliklinisch','creatinine - spoed','ijzer','natrium - vlamfotometrisch - spoed','kalium vlamfotometrisch - spoed','foliumzuur mbv radioisotopen','vitamine b12 mbv chemieluminescentie','bloedgroep abo en rhesusfactor','rhesusfactor d - centrifugeermethode - e','haemoglobine foto-electrisch - spoed','trombocyten tellen - spoed','reticulocyten tellen mbv facscan','cde fenotypering','bloedgroepantigenen andere dan abo-rhesu','kruisproef volledig -drie methoden-','leucocyten tellen -electronisch- spoed','crp c-reactief proteine - spoed','screening antistoffen erytrocyten','transferrine mbv ics','190101 bovenreg.toesl.  a101','gefiltreerd erytrocytenconcentraat','echo nieren-urinewegen','interc.consult klinisch  anesthesie','ureum - spoed','sgpt alat kinetisch - spoed','sgot asat kinetisch - spoed','international normalised ratio mbv tromb','dagverpleging - alle spec.beh.kind.-rev.','190035 dagverpleging    a007','urinezuur met uricase u.v. spectrofotome','melkzuur dehydrogenase - spoed','echo abdomen','sediment','klinische kaart - inwendige geneeskunde','inwend.geneesk.   aanv.kaart kosten-out','squamous cell carcinoma mbv eia','ct abdomen','ca-125 mbv meia','telefonisch consult','patient niet verschenen radiologie','homocystine chromatografisch','cytologisch onderzoek - ectocervix -','bilirubine -geconjugeerd','bilirubine - totaal','glucose','alkalische fosfatase -kinetisch-','gammaglutamyltranspeptidase','cea - tumormarker mbv meia','calcium','albumine','thorax','190204 klasse 3a        a204','buik        - stag.lap-tumorred.-omentec','immunopathologisch onderzoek','cytologisch onderzoek - ascites -','histologisch onderzoek - grote resectiep','glucose - spoed','calcium - spoed','melkzuur enzymatisch - spoed','o2-saturatie','ct thorax','verlosk.-gynaec.  aanv.kaart kosten-out','hematocriet mbv centrifuge','totaal eiwit','vulva       - rad.vulvectomie - opp.en d','microscopisch onderzoek - gekleurd en on','coupe ter inzage','mri bekken','skelet - scintigrafie totale lichaam','histologisch onderzoek - biopten nno','cito histologisch onderzoek','fosfaat','ct bovenbuik','blaas       - uretrocystoscopie nno','vagina      - toucher onder anesthesie','wervelkolom - lumbaal','ct hersenen','differentiatie leukocyten - handmatig','anesthesie  - bij behandeling met radium','albumine - spoed','bilirubine kwantitatief totaal of direct','alfa amylase - spoed','chloride - spoed','fosfaat - spoed','alkalische fosfatase - kinetisch - spoed','totaal eiwit colorimetrisch - spoed','kreatinine fosfokinase - spoed','hematocriet - spoed','gamma glutamyltranspeptidase - spoed','ferritine kwn mbv chemieluminescentie','erythrocyten tellen -electronisch- spoed','osmolaliteit - spoed','bezinkingssnelheid foto-elektrisch','magnesium  -aas - spoed','mri wervelkolom - cervicaal','mri wervelkolom - thoracaal','mri wervelkolom - lumbaal','cortisol immunofluorimetrisch','tsh enzym-immunologisch','cortisol','vaten       - venapunctie','acth mbv radioisotopen','e.c.g.      - elektrocardiografie','lithium','vrij t4 mbv ria -ft 4','1e consult      bezoek','lymfeklier - scint sentinel node dynamis','lymfeklier - scint sentinel node vervolg','lymfeklier - scint sentinel node met pro','vulva       - radicale vulvect. znd ingu','vriescoupe','methemoglobine - sulfhemoglobine elk','bicarbonaat','co-hb kwn.','actuele ph - pco2 - stand.bicarbonaat','lithium - spoed','verlosk.-gynaec.   jaarkaart kosten-out','histologisch onderzoek - kleine resectie','vrw.gesl.org- biopsie-punctie-cytologie','buikoverzicht','klinische kaart - anesthesie','arterie     - punctie tbv.verblijfsnaald','thorax - op zaal','protrombinetijd -quick owren of modif.-','heparinebepaling - kaoline - cefalinetij','interc.consult klinisch  longziekten','dieet nno','buik        - primaire stageringsoperati','anti-hiv mbv elisa','patient niet verschenen ngiv- no show -','infuus      - inbrengen','cytologisch onderzoek - vagina -','tumor - petscan totale lichaam','blaas       - cystoscopie met biopsie','cytologisch onderzoek - lymfeklierpuncti','diagnostische punctie onder echocontrole','echo schouder - m-z bovenarm','cytologisch onderzoek - pancreaspunctie','echo bovenbuik','hyperthermie','hele bovenbeen met onderbeen en voet','magnesium  -aas','bekken','bekken - liggend','fdp - dimeer - immunologisch','ct art. en ven.pulmonales','paclitaxel','behandeltijd - eenheid t1 - megavolt','chloride','bovenbeen','onderbeen','natrium vlamfotometrisch - spoed','cefalinetijd - coagulatie','protrombinetijd','vulva       - incisie - overige','echografie  - blaas - extern','punctie tbv cytologisch onderzoek door p','vulva       - vulvectomie - liesblok','microscopisch onderzoek  - elektronenmic','buik        - stageringslaparotom.-oment','mammografie thoraxwand','echo mamma','vrw.gesl.org- curettage gefractioneerd','vulva       - biopsie-punctie-cytologie','uterus      - hysteroscopie','uterus      - extirp.abd.rad. - lymfaden','toonaudiometrie - audiologisch centrum','hoogfrequente audiometrie - audiologisch','tympanometrie - standaard - audiologisch','patienten kontakt audiologie korter dan','anesthesie  - bij spec.onderz.en verr.ge','bekken - staand','cytologisch onderzoek - nierpunctie -','doorlichting zonder opname thorax','eiwit beperkt - mineralen beperkt','pyelografie antegraad via drain','verwisselen van nefrostomie drain','lipase','hemoglobine a1','190055 zware dagverpleging a009','mri abdomen','directe antiglobulinetest - coombs','antistoffen tegen erys - specificiteit -','ident. irregulaire a.s. tegen erys na po','190034 afwezigheidsdag  a006','ligase chain reaction -lcr-','uterus      - intra-uterine device inbre','mri bijnier','melkzuur enzymatisch','blaas       - catheter a demeure - inbre','betasubunit bep. mbv. ria hcg','sikkelcel','alfa-1-foetoproteine','cholesterol totaal','triglyceriden - enzymatisch','cholesterol hdl','afereseplasma gesplitst fq1 en fq2','ca 15.3 - tumormarker mbv meia','ca-19.9 tumormarker','behandeltijd - eenheid t4 - megavolt','ct bekken','heup','ct wervelkolom - lumbaal','bot - densitometrie lwk','bot - densitometrie femurhals','titratie directe antiglobulinetest - coo','vrw.gesl.org- exc.destruct.path.afwijkin','analgesie   - epiduraal door anesthesist','urinewegen  - urodynamisch onderzoek-vij','uterus      - extirpatie abdominaal radi','maag - ontledig.-scint vast-vloeib.voeds','darm - colonpassage scint.comp.vervolg','mri heup','echo onderbuik','creatine fosfokinase mb - spoed','troponine-t mbv elisa','behandeltijd - eenheid t2 - megavolt','ovarium     - debulking ovariumcarcinoom','cervix      - lisexcisie portio - diathe','uterus      - extirpatie abdominaal tota','hart        - doppler echocardiografie 2','uterus      - carc. cervix vlgs. werthei','hematologie - ontstek.scint.tot.lichaam','ovarium     - adnex-extirpatie dmv lapar','pyelografie --wo niertransplantaat--','longfunctie - co2 - capnografie','amylase','ammoniak','fibrinogeen','trombocyten - leuko verwijderd - 5 donor','haptoglobine','trombinetijd','celkweek en isolatie','determinatie en orient. typering -dmv ce','buik        - proeflaparotomie','blaas       - suprapubische katheter inb','dikke darm  - sigmoidoscopie met flexibe','vulva       - vulvectomie zonder lieskli','interc.consult klinisch  radiotherapie','zwangerschapsreactie - spoed','creatinefosfokinase -kinetisch-','ethanol -enzymatisch of gaschromatografi','colon inloop','inwend.geneesk.    jaarkaart kosten-out','vitamine d3','onderzoek dunne darm m-z duodenum','wervelkolom - cervicaal','ammoniak - spoed','fenytoine kwantitatief','antitrombine iii mbv chromogeen-n-substr','mri grote hersenen','wervelkolom - thoracaal','wervelkolom - thoracolumbale overgang','toegangschir- c.v.d.-katheter perifeer b','erytrocyten leukocyten verwijderd bestra','interc.consult klinisch  neurologie','echogeleide als ass. bij punctie-biopsie','pijnbestrijd- aanleggen patient controll','ct a.pulmonalis','cytologisch onderzoek - liquor cerebrosp','erytrocyten','buffy coat','klinisch tarief','cytologisch onderzoek - diversen -','totaal t4 - thyroxine immunofluorimetris','vagina      - biopsie-punctie-cytologie','blaas       - cystoscopie','sinus','clostridium elisa-test','kalium vlamfotometrisch','perif.vaten - duplexonderzoek - veneus','dikke darm  - appendectomie nno','interc.consult klinisch  chirurgie','voet','cytologisch onderzoek - urine niercyste','schoudergewricht','chloride -ionselectief-','antinucleaire factor -anf-','bovenarm','vagina      - scopie incl.evt.vulvabiops','vagina      - overige excisie pathologis','vulva       - ruime lokale excisie van a','nieren - renografie lasix','parathormoon mbv radioisotopen','orthopantomogram \'panoramixopname\' gebit','echo hals --waaronder schildklier--','mec-a-bepaling voor mrsa mbv pcr','buik        - punctie ascites ontlastend','uterus      - extirpatie - abdominaal me','doxorubicine liposomal --caelyx--','ct lever en galwegen','onderzoek van elders niet beoordeeld','cervix      - conisatie','morfometrie','bloedgroepbep. anders dan abo per bloedg','uitsluiten irregulaire a.s. tegen erys d','punctie cyste tumor ed. door radioloog o','pyelografie antegraad via punctie','vitamine b6','vitamine e -hplc-methode','vitamine b1 - thiamine','interc.consult vervolg klinisch  interne','intensive care ligdag - 1e tot en met 5e','anesthesie  - bij vervoer beademde patie','creatine fosfokinase isoenzym ckmb','cvvhd       --continue veno-veneuze hemo','intensive care ligdag - 6e tot en met 14','uterus      - curettage - diagnostisch','cervix      - curettage','buik        - stag.laparotom.-omentect.-','echo tractus urogenitalis-bekken','eiwit -colorimetrisch- totaal','microalbumine immunonefelometrisch','ct hals','aluminium','erytrocyten tellen - elektronisch','duplexscan - venen been','fsh enzym-immunologisch','lh mbv radioisotopen','progesteron mbv radioisotopen','prolactine mbv eia','oestradiol mbv radioisotopen -ria','vrw.gesl.org- plastische operatie van vu','testosteron mbv radioisotopen','sex-hormone-binding globuline mbv ria','vagina      - tension-free vaginal tape','vagina      - colpocleisis lefort','gemcitabine','buikoverzicht - op zaal','echografie-doppler vena cava inferior en','echo been','vitamine b2','vitamine b3','vitamine a','beademing - anesthesie - eerste dag','lymf.syst.  - stageringslymfadenectomie','widal agglutinatie - yersinia 3','widal agglutinatie - yersinia 9','dikke darm  - colonoscopie met biopsie','darm        - katheteriseren stoma','interc.consult vervolg klinisch  anesthe','hart - functie gated bloodpool rust','echografie a vue ivw zwangerschap - met','onderzoek via percutane veneuze katheter','echografie-doppler v.jugularis','ven. d.s.a. vena cava inferior','anti-hepatitis-c-virus mbv elisa','beademing - anesthesie - tweede t-m vijf','klinische kaart - verloskunde en gynaeco','vulva       - partiele vulvectomie','vrw.gesl.org- adnex-extirpatie dmv lapar','zwangerschapsreactie','rib-sternum','interc.consult klinisch  interne ziekten','brachytherapie - interstitieel - bijzond','osmolaliteit','digoxine','hepatitis-c-virus roche -hcvr- mbv pcr','cervix      - portio - lasercoagulatie','vagina      - scopie excl.weefselwegname','cytologisch onderzoek - pleuravocht -','huidplastiek- z-plastiek huid hoofd en h','ct totale lichaam','pyelografie retrograad','echo pancreas en milt','brain natriuretic peptide -bnp- mbv immu','duodenum    - oesofagogastroduodenoscopi','interc.consult klinisch  urologie','darm - colonpassage scint.comp.statisch','echo en rontgen als ass. bij drainage ab','gentamycine kwn - immuno-assay','echo thorax','ro-gel. als ass. bij punktie - biopsie t','onderz. naar aanwezigheid van koude anti','geb. antistoffen tegen erys - dir.coombs','geb. antistoffen tegen erys - dir. coomb','ovarium     - redebulking ovarium carcin','ovarium     - excisie cyste via laparosc','behandelen  - in hyperpressietank','perif.vaten - doppler-onderz.oppervl.ven','perif.vaten - duplexscan arterieel cruro','buik        - second-look-oper.stagering','elutie van auto-antistoffen tegen erytro','bloedgroepbepaling anders dan abo  kidd','buik        - excisie patholog.afwijking','17-hydroxyprogesteron mbv ria','delta-4-androsteendion','dhea-sulfaat met extractie mbv ria','enkel','vrw.gesl.org- aspiratie of microcurett.-','bronchus    - bronch.toilet door verpl.-','immunoglobuline a                    iga','heparine anti-xa','ro-gel. als ass. bij drainage nier-nefro','echo nier','occult bloed - kwalitatief-','echo buikwand','cytologisch onderzoek - buiktumorpunctie','buik        - omentumplastiek','ct nier','cytologisch onderzoek - leverpunctie -','echogeleide als ass.bij punctie-biopsie','hart        - resuscitatie','uterus      - extirpatie abdom.totaal me','vrw.gesl.org- adhesiolysis ovarium plus','cytologisch onderzoek - vulva -','ct retroperitoneum','anti-hav ig of igg-elisa of ria- in een','antist.tegen hepatitis-b surface antigee','hepatitis-b-anti-core igm','buik        - littekenbreuk - plastische','vagina      - operatie vesicovaginale fi','vulva       - hechten van vulva','mycobacterium mbv pcr','huid        - transcutane po2 en pco2 me','glucosebepaling -kwantitatief -','ct-gel. als ass. bij punctie - biopsie i','cytologisch onderzoek - endocervix -','oestron mbv radioisotopen','lymf.syst.  - lymfadenectomie dmv laparo','cytostatica - infuustherapie','echogeleide als ass. bij drainage intrap','rituximab','vagina      - vaginotomie','echografie a vue ivm zwangerschap - rout','blaas       - voorste en achterste exent','ureter - splintografie cq opspuiten uret','blaas       - bladder wash-out','duodenum    - inbrengen voedingssonde in','e.m.g.      - elektromyografisch onderzo','knie - m-z onderbeen','ct arterien hersenen','igg antistoffen tegen heparine-pf4-compl','abortus     - overige technieken','echografie  - transuretraal prostaat','eiwitspectrum -elektroforese kwn. bep. f','immunoglobuline g                    igg','immunoglobuline m                    igm','paraproteine typering mbv monospecifieke','eiwitfractionering mbv ief','immunofixatie','immunoforese na concentratie','drainageprocedure onder rontgencontrole','antigeendetectie - direct preparaat - hs','antigeendetectie - direct preparaat - vz','niet declarabele dagverpleging-bv klin.p','interc.consult klinisch  reumatologie','anti-cytomegalovirus ig of igg -elisa','igm-as. tegen cmv virus','anti-epstein-barr vca igm mbv elisa','anti-epstein-barr na igg mbv elisa','anti-epstein-barr vca igg mbv elisa','complementcomponent c3 kwn.','complementcomponent c4 kwn .','dna - farr-assay mbv ria','legionella antigeen sneltest','complementdeficientie ond.bij afwezigh.a','longfunctie - spirometrie','longfunctie - stroomvolume curve','sacrum-os coccygis-s.i.gewrichten','proteine-c-activiteit','igm anticardiolipine mbv elisa','igg anticardiolipine mbv elisa','antist. specifiek mbv elisa histonen of','stollingsfactor - viii - activiteit','anticoagulans lupus mbv elisa','proteine-s-antigeen -totaal','proteine-s-antigeen -vrij','factor ii mutatie mbv pcr','factor v leiden -r506q- mbv dna-test','lymf.syst.  - lymfekliertoilet diep - li','tuba uterina- excisie parovariale cyste','echo bovenbeen','ph-meting','spraakaudiometrie  -  standaard - audiol','mandibula-kaakgewricht','mri orbita','high sensitive crp -hscrp- nefelometrisc','echo arteriae carotis sinistra','echo arteriae carotis dextra','bloedstollingsfactor xi - activiteit','bloedstollingsfactor xii - activiteit','bloedstollingsfactor ix - coagulatie','uterus      - excisie-destructie path.af','hyperthermie behandeling - h-1 -','inbrengen voedings- of lavagesonde duode','echo pleura','pleura      - punctie - diagnostisch','alfa-amylase','vancomycine kwn - immuno-assay','interc.consult vervolg klinisch  chirurg','echografie-doppler venen been','mictie-cysto-uretrografie','totale gebitsstatus','vrw.gesl.org- adhesiolysis ovaria-tubae','ct pancreas en milt','doorlichting zonder opname buik','echogeleide als ass. bij punctie abdomen','cytologisch onderzoek tbv bevolkingsonde','bloedstollingsfactor v - kwantitatief -','bloedstollingsfactor vii - coagulatie','von willebrand profiel mbv protease','cytologisch onderzoek - schildklier  pun','uterus      - extirpatie vaginaal totaal','nucleair onderzoek van elders','unknown','peritoneum  - omentectomie','vulva       - therapeutische punctie aan','urethra     - dilatatie meatus','antist. tegen acetylcholinereceptoren','aspect supernatant','infuus      - bloed','thorax      - zuigdrainage behandeling a','ovarium     - ovariopexie -excl.torsie-','prostaatspecifiek antigeen -psa- mbv imm','antist. tegen hiv-e of hiv-c','direct ondz. naar rs-virus antigeen mbv','para-influenza-1-virus mbv pcr','humaan metapneumovirus kwl. mbv dna-rna-','antigeendetectie mbv ift -influenza-a','antigeendetectie mbv ift -influenza-b','buik        - laparoscopie - diagnostisc','wormeieren - uitgebreid -vervolg-','zink','vrw.gesl.org- operatie van clitoris nno','opname volgens barsonie','lymf.syst.  - lymfeklierextirpatie para-','uterus      - extirp.abd.vag.rad.- lymfa','buikoverzicht - staand','dunne darm  - mech.ileus-adhes.streng.-i','cytologisch onderzoek - hersenen -','anti-rubella igg -elisa - in een monster','anti-chlamydia iga mbv elisa','anti-chlamydia igg mbv ind.ift','buik        - adhesiolyse en biopsie dmv','endoretrograde cholangiografie','cystografie retrograad','doorlichting bij niersteenvergruizing','wisselen drain onder rontgencontrole','kleurproef sabin-feldman - leishmania','toxoplasmose mbv eia toxo-igg','toxoplasmose mbv eia toxo-igm','anti-rubella igm -elisa','cbr -coxsackie-b-900-virus','anti-parvovirus -ig of igg antist. ind.','anti-parvovirus igm mbv ind.ift','tuba uterina- aanbrengen clips tijdens l','partus      - klin.of poliklin.met voorb','sect.caesar.- met voorbehandeling en kra','mri tractus urogenitalis','opspuiten hickman-katheter','ro-gel. als ass. bij punctie - biopsie l','lymf.syst.  - regionale lymfeklierdissec','339849d toegangschir- c.v.d.-catheter v.','hemoglobinen -abnormaal incl. hba2','igg-index','albumineratio','immuno-elektroforese liquor na concentra','huid        - verwijderen retentie cyste','ro-gel. als ass. bij inbrengen endoproth','lever       - excisie tumor of metastase','lymf.syst.  - sentinel node procedure li','gipskamer-ok - gebruik','standaard materiaal n.n.o.','zwachtelverband sling - aanleggen','partiele gebitsstatus','interc.consult poliklinisch','vrw.gesl.org- second-look-oper.ovariumca','oesofagografie - oraal','1-25-oh2-vitamine d mbv hplc','perif.vaten - verwijderen port-a-cath','perif.vaten - inbrengen port-a-cath syst','rontgenonderzoek onvolledig','mri art. en ven.abdomen','vagina      - extirpatie van vagina met','neuronspecifiek enolase','ct knie','tuba uterina- reanastomose na sterilisat','obductie','anesthesie  - epiduraal met bloedpatch','dikke darm  - resectie sigmoid met prima','dikke darm  - appendectomie acuut via me','totaal t3 - trijoodthyronine mbv radiois','t3-uptake latente bindingscap. geb. tbg','echo bekken inhoud muv anatomiecode 83 t','uterus      - tot. laparoscopische hyste','plaatsen van v.cava filter','eerste hulp - niet seh afdeling - elders','buik        - second-look-oper.ovariumca','itraconazol mbv hplc','bepaling hepatitis-be-antigeen','bepaling antist. hepatitis-be-antigeen']
    seq_act_features = ['1e consult poliklinisch', 'administratief tarief       - eerste pol',
                        'verlosk.-gynaec. korte kaart kosten-out', 'echografie  - genitalia interna',
                        'simulator - gebruik voor aanvang megavol', 'behandeltijd - eenheid t3 - megavolt',
                        'teletherapie - megavolt fotonen bestrali', 'aanname laboratoriumonderzoek', 'ureum',
                        'hemoglobine foto-elektrisch', 'creatinine', 'natrium vlamfotometrisch',
                        'kalium potentiometrisch', 'leukocyten tellen elektronisch',
                        'trombocyten tellen - elektronisch', 'ordertarief', 'ligdagen - alle spec.beh.kinderg.-reval.',
                        'sediment - spoed', 'bacteriologisch onderzoek met kweek -nie',
                        'resistentiebepalingen - 5 bepalingen', 'hepatitis-b surface antigeen confirmatie',
                        'urine onderzoek - kwalitatief', '190021 klinische opname a002', '190205 klasse 3b        a205',
                        'sgot - asat kinetisch', 'sgpt - alat kinetisch', 'melkzuurdehydrogenase -ldh- kinetisch',
                        'differentiele telling automatisch', 'trombotest', 'crp c-reactief proteine',
                        'brachytherapie - interstitieel - intensi', 'inwend.geneesk.  korte kaart kosten-out',
                        'vervolgconsult poliklinisch', 'creatinine - spoed', 'ijzer',
                        'natrium - vlamfotometrisch - spoed', 'kalium vlamfotometrisch - spoed',
                        'foliumzuur mbv radioisotopen', 'vitamine b12 mbv chemieluminescentie',
                        'bloedgroep abo en rhesusfactor', 'rhesusfactor d - centrifugeermethode - e',
                        'haemoglobine foto-electrisch - spoed', 'trombocyten tellen - spoed',
                        'reticulocyten tellen mbv facscan', 'cde fenotypering',
                        'bloedgroepantigenen andere dan abo-rhesu', 'kruisproef volledig -drie methoden-',
                        'leucocyten tellen -electronisch- spoed', 'crp c-reactief proteine - spoed',
                        'screening antistoffen erytrocyten', 'transferrine mbv ics', '190101 bovenreg.toesl.  a101',
                        'gefiltreerd erytrocytenconcentraat', 'echo nieren-urinewegen',
                        'interc.consult klinisch  anesthesie', 'ureum - spoed', 'sgpt alat kinetisch - spoed',
                        'sgot asat kinetisch - spoed', 'international normalised ratio mbv tromb',
                        'dagverpleging - alle spec.beh.kind.-rev.', '190035 dagverpleging    a007',
                        'urinezuur met uricase u.v. spectrofotome', 'melkzuur dehydrogenase - spoed', 'echo abdomen',
                        'sediment', 'klinische kaart - inwendige geneeskunde',
                        'inwend.geneesk.   aanv.kaart kosten-out', 'squamous cell carcinoma mbv eia', 'ct abdomen',
                        'ca-125 mbv meia', 'telefonisch consult', 'patient niet verschenen radiologie',
                        'homocystine chromatografisch', 'cytologisch onderzoek - ectocervix -',
                        'bilirubine -geconjugeerd', 'bilirubine - totaal', 'glucose',
                        'alkalische fosfatase -kinetisch-', 'gammaglutamyltranspeptidase', 'cea - tumormarker mbv meia',
                        'calcium', 'albumine', 'thorax', '190204 klasse 3a        a204',
                        'buik        - stag.lap-tumorred.-omentec', 'immunopathologisch onderzoek',
                        'cytologisch onderzoek - ascites -', 'histologisch onderzoek - grote resectiep',
                        'glucose - spoed', 'calcium - spoed', 'melkzuur enzymatisch - spoed', 'o2-saturatie',
                        'ct thorax', 'verlosk.-gynaec.  aanv.kaart kosten-out', 'hematocriet mbv centrifuge',
                        'totaal eiwit', 'vulva       - rad.vulvectomie - opp.en d',
                        'microscopisch onderzoek - gekleurd en on', 'coupe ter inzage', 'mri bekken',
                        'skelet - scintigrafie totale lichaam', 'e.c.g.      - elektrocardiografie', 'lithium',
                        'tsh enzym-immunologisch', 'vrij t4 mbv ria -ft 4', '1e consult      bezoek',
                        'lymfeklier - scint sentinel node dynamis', 'lymfeklier - scint sentinel node vervolg',
                        'lymfeklier - scint sentinel node met pro', 'vulva       - radicale vulvect. znd ingu',
                        'vriescoupe', 'methemoglobine - sulfhemoglobine elk', 'bicarbonaat', 'co-hb kwn.',
                        'actuele ph - pco2 - stand.bicarbonaat', 'lithium - spoed',
                        'histologisch onderzoek - biopten nno', 'verlosk.-gynaec.   jaarkaart kosten-out',
                        'histologisch onderzoek - kleine resectie', 'cito histologisch onderzoek',
                        'vrw.gesl.org- biopsie-punctie-cytologie', 'blaas       - uretrocystoscopie nno',
                        'vagina      - toucher onder anesthesie', 'bezinkingssnelheid foto-elektrisch', 'buikoverzicht',
                        'klinische kaart - anesthesie', 'anesthesie  - bij behandeling met radium',
                        'arterie     - punctie tbv.verblijfsnaald', 'thorax - op zaal', 'chloride - spoed',
                        'fosfaat - spoed', 'protrombinetijd -quick owren of modif.-',
                        'heparinebepaling - kaoline - cefalinetij', 'albumine - spoed',
                        'interc.consult klinisch  longziekten', 'dieet nno', 'buik        - primaire stageringsoperati',
                        'behandeltijd - eenheid t1 - megavolt', 'fosfaat', 'chloride', 'magnesium  -aas',
                        'bekken - liggend', 'bovenbeen', 'onderbeen', 'alkalische fosfatase - kinetisch - spoed',
                        'magnesium  -aas - spoed', 'buik        - stageringslaparotom.-oment',
                        'infuus      - inbrengen', 'echografie  - blaas - extern', 'mammografie thoraxwand',
                        'echo mamma', 'vrw.gesl.org- curettage gefractioneerd',
                        'vulva       - biopsie-punctie-cytologie', 'uterus      - hysteroscopie',
                        'ligase chain reaction -lcr-', 'uterus      - intra-uterine device inbre',
                        'cytologisch onderzoek - vagina -', 'mri bijnier', 'melkzuur enzymatisch',
                        'behandeltijd - eenheid t4 - megavolt', 'ct bekken', 'bekken - staand', 'heup',
                        'ct wervelkolom - lumbaal', 'bot - densitometrie lwk', 'bot - densitometrie femurhals',
                        'alfa-1-foetoproteine', 'vrw.gesl.org- exc.destruct.path.afwijkin', 'mri abdomen',
                        'analgesie   - epiduraal door anesthesist', 'eiwit beperkt - mineralen beperkt',
                        'urinewegen  - urodynamisch onderzoek-vij', 'uterus      - extirpatie abdominaal radi',
                        'maag - ontledig.-scint vast-vloeib.voeds', 'darm - colonpassage scint.comp.vervolg',
                        'mri wervelkolom - lumbaal', 'mri heup', 'paclitaxel', 'kreatinine fosfokinase - spoed',
                        'creatine fosfokinase mb - spoed', 'troponine-t mbv elisa',
                        'behandeltijd - eenheid t2 - megavolt', 'ovarium     - debulking ovariumcarcinoom',
                        'cervix      - lisexcisie portio - diathe', 'hematocriet - spoed',
                        'uterus      - extirpatie abdominaal tota', 'cefalinetijd - coagulatie', 'protrombinetijd',
                        'cholesterol totaal', 'triglyceriden - enzymatisch', 'cholesterol hdl',
                        'hart        - doppler echocardiografie 2', 'uterus      - carc. cervix vlgs. werthei',
                        'hematologie - ontstek.scint.tot.lichaam', 'ovarium     - adnex-extirpatie dmv lapar',
                        'gamma glutamyltranspeptidase - spoed', 'pyelografie --wo niertransplantaat--',
                        'blaas       - suprapubische katheter inb', 'dikke darm  - sigmoidoscopie met flexibe',
                        'vulva       - vulvectomie zonder lieskli', 'interc.consult klinisch  radiotherapie',
                        'erytrocyten leukocyten verwijderd bestra', 'differentiatie leukocyten - handmatig',
                        'interc.consult klinisch  neurologie', 'klinisch tarief', 'cytologisch onderzoek - diversen -',
                        'totaal t4 - thyroxine immunofluorimetris', 'vagina      - biopsie-punctie-cytologie',
                        'blaas       - cystoscopie', 'perif.vaten - duplexonderzoek - veneus', 'clostridium elisa-test',
                        'dikke darm  - appendectomie nno', 'bilirubine kwantitatief totaal of direct',
                        'punctie tbv cytologisch onderzoek door p', 'cytologisch onderzoek - urine niercyste',
                        'schoudergewricht', 'bovenarm', 'colon inloop', 'anesthesie  - bij spec.onderz.en verr.ge',
                        'vagina      - scopie incl.evt.vulvabiops', 'vagina      - overige excisie pathologis',
                        'vulva       - ruime lokale excisie van a', 'nieren - renografie lasix',
                        'vulva       - vulvectomie - liesblok', 'totaal eiwit colorimetrisch - spoed', 'echo bovenbuik',
                        'uterus      - extirpatie - abdominaal me', 'ct lever en galwegen',
                        'onderzoek van elders niet beoordeeld', 'cervix      - conisatie', 'morfometrie',
                        'directe antiglobulinetest - coombs', 'bloedgroepbep. anders dan abo per bloedg',
                        'antistoffen tegen erys - specificiteit -', 'ident. irregulaire a.s. tegen erys na po',
                        'uitsluiten irregulaire a.s. tegen erys d', 'cytologisch onderzoek - lymfeklierpuncti',
                        'punctie cyste tumor ed. door radioloog o', 'pyelografie antegraad via punctie',
                        'uterus      - curettage - diagnostisch', 'cervix      - curettage',
                        'buik        - stag.laparotom.-omentect.-', 'echo tractus urogenitalis-bekken',
                        'kalium vlamfotometrisch', 'eiwit -colorimetrisch- totaal',
                        'microalbumine immunonefelometrisch', 'ct hals', 'aluminium',
                        'erytrocyten tellen - elektronisch', 'betasubunit bep. mbv. ria hcg', 'fsh enzym-immunologisch',
                        'lh mbv radioisotopen', 'progesteron mbv radioisotopen', 'prolactine mbv eia',
                        'oestradiol mbv radioisotopen -ria', 'vrw.gesl.org- plastische operatie van vu',
                        'testosteron mbv radioisotopen', 'sex-hormone-binding globuline mbv ria', 'ca-19.9 tumormarker',
                        'diagnostische punctie onder echocontrole', 'echo hals --waaronder schildklier--',
                        'vagina      - tension-free vaginal tape', 'vagina      - colpocleisis lefort', 'ct bovenbuik',
                        'natrium vlamfotometrisch - spoed', 'lymf.syst.  - stageringslymfadenectomie',
                        'widal agglutinatie - yersinia 3', 'widal agglutinatie - yersinia 9', 'amylase',
                        'dikke darm  - colonoscopie met biopsie', 'inwend.geneesk.    jaarkaart kosten-out',
                        'hemoglobine a1', 'wervelkolom - lumbaal', 'tumor - petscan totale lichaam',
                        'klinische kaart - verloskunde en gynaeco', 'vulva       - partiele vulvectomie',
                        'mri grote hersenen', 'buik        - proeflaparotomie',
                        'vrw.gesl.org- adnex-extirpatie dmv lapar', 'zwangerschapsreactie',
                        'anti-hepatitis-c-virus mbv elisa', 'hepatitis-c-virus roche -hcvr- mbv pcr',
                        'anti-hiv mbv elisa', 'alfa amylase - spoed', 'cervix      - portio - lasercoagulatie',
                        'vagina      - scopie excl.weefselwegname', '190034 afwezigheidsdag  a006',
                        'interc.consult klinisch  chirurgie', 'ca 15.3 - tumormarker mbv meia',
                        'cytologisch onderzoek - pleuravocht -', 'huidplastiek- z-plastiek huid hoofd en h',
                        'pyelografie antegraad via drain', 'ct totale lichaam', 'pyelografie retrograad',
                        'echo pancreas en milt', 'brain natriuretic peptide -bnp- mbv immu',
                        'ferritine kwn mbv chemieluminescentie', 'chloride -ionselectief-',
                        'onderz. naar aanwezigheid van koude anti', 'geb. antistoffen tegen erys - dir.coombs',
                        'geb. antistoffen tegen erys - dir. coomb', 'ovarium     - redebulking ovarium carcin',
                        'creatinefosfokinase -kinetisch-', 'lipase', 'duplexscan - venen been',
                        'toonaudiometrie - audiologisch centrum', 'hoogfrequente audiometrie - audiologisch',
                        'patienten kontakt audiologie korter dan', 'titratie directe antiglobulinetest - coo',
                        'elutie van auto-antistoffen tegen erytro', 'bloedgroepbepaling anders dan abo  kidd',
                        'echo onderbuik', 'buik        - excisie patholog.afwijking', '17-hydroxyprogesteron mbv ria',
                        'delta-4-androsteendion', 'dhea-sulfaat met extractie mbv ria',
                        'vrw.gesl.org- aspiratie of microcurett.-', 'interc.consult klinisch  interne ziekten',
                        'digoxine', 'bronchus    - bronch.toilet door verpl.-', 'ct hersenen',
                        'uterus      - extirp.abd.rad. - lymfaden', 'immunoglobuline a                    iga',
                        'vitamine b1 - thiamine', 'hart        - resuscitatie',
                        'uterus      - extirpatie abdom.totaal me', 'longfunctie - co2 - capnografie',
                        'vrw.gesl.org- adhesiolysis ovarium plus', 'cytologisch onderzoek - vulva -',
                        'ct retroperitoneum', 'anti-hav ig of igg-elisa of ria- in een',
                        'antist.tegen hepatitis-b surface antigee', 'hepatitis-b-anti-core igm',
                        'echogeleide als ass.bij punctie-biopsie', 'ethanol -enzymatisch of gaschromatografi',
                        'buik        - littekenbreuk - plastische', 'zwangerschapsreactie - spoed', 'hyperthermie',
                        'vagina      - operatie vesicovaginale fi', 'vulva       - hechten van vulva',
                        'ct-gel. als ass. bij punctie - biopsie i', 'buik        - punctie ascites ontlastend',
                        'antitrombine iii mbv chromogeen-n-substr', 'cytologisch onderzoek - endocervix -',
                        'oestron mbv radioisotopen', 'cytologisch onderzoek - leverpunctie -', 'fibrinogeen',
                        'vagina      - vaginotomie', 'echografie a vue ivm zwangerschap - rout',
                        'echografie a vue ivw zwangerschap - met', 'echo been', 'fdp - dimeer - immunologisch',
                        'ct a.pulmonalis', 'proteine-c-activiteit', 'igm anticardiolipine mbv elisa',
                        'igg anticardiolipine mbv elisa', 'antist. specifiek mbv elisa histonen of', 'trombinetijd',
                        'stollingsfactor - viii - activiteit', 'anticoagulans lupus mbv elisa',
                        'proteine-s-antigeen -totaal', 'proteine-s-antigeen -vrij', 'factor ii mutatie mbv pcr',
                        'factor v leiden -r506q- mbv dna-test', 'tuba uterina- excisie parovariale cyste',
                        'echo bovenbeen', 'mri wervelkolom - thoracaal', 'pijnbestrijd- aanleggen patient controll',
                        'spraakaudiometrie  -  standaard - audiol', 'mandibula-kaakgewricht', 'mri orbita',
                        'mri wervelkolom - cervicaal', 'high sensitive crp -hscrp- nefelometrisc',
                        'echo arteriae carotis sinistra', 'echo arteriae carotis dextra',
                        'onderzoek via percutane veneuze katheter', 'ven. d.s.a. vena cava inferior',
                        'creatine fosfokinase isoenzym ckmb', 'haptoglobine',
                        'uterus      - excisie-destructie path.af', 'rib-sternum', 'hyperthermie behandeling - h-1 -',
                        'echografie-doppler venen been', 'afereseplasma gesplitst fq1 en fq2',
                        'mictie-cysto-uretrografie', 'ct nier', 'totale gebitsstatus',
                        'orthopantomogram \'panoramixopname\' gebit', 'vrw.gesl.org- adhesiolysis ovaria-tubae',
                        'ct pancreas en milt', 'doorlichting zonder opname buik',
                        'cytologisch onderzoek tbv bevolkingsonde', 'blaas       - cystoscopie met biopsie',
                        'vaten       - venapunctie', 'uterus      - extirpatie vaginaal totaal',
                        'nucleair onderzoek van elders', 'unknown', 'peritoneum  - omentectomie',
                        'vulva       - therapeutische punctie aan', 'urethra     - dilatatie meatus',
                        'infuus      - bloed', 'cytologisch onderzoek - buiktumorpunctie',
                        'echogeleide als ass. bij punctie abdomen', 'echo schouder - m-z bovenarm',
                        'prostaatspecifiek antigeen -psa- mbv imm', 'buik        - laparoscopie - diagnostisc',
                        'vrw.gesl.org- operatie van clitoris nno', 'opname volgens barsonie',
                        'lymf.syst.  - lymfeklierextirpatie para-', 'uterus      - extirp.abd.vag.rad.- lymfa',
                        'anti-rubella igg -elisa - in een monster', 'anti-chlamydia iga mbv elisa',
                        'anti-chlamydia igg mbv ind.ift', 'buik        - adhesiolyse en biopsie dmv',
                        'endoretrograde cholangiografie', 'cystografie retrograad', 'mri tractus urogenitalis',
                        'longfunctie - spirometrie', 'longfunctie - stroomvolume curve',
                        'ro-gel. als ass. bij punctie - biopsie l', '339849d toegangschir- c.v.d.-catheter v.',
                        'trombocyten - leuko verwijderd - 5 donor', 'hemoglobinen -abnormaal incl. hba2',
                        'cytologisch onderzoek - liquor cerebrosp', 'immunoglobuline m                    igm',
                        'igg-index', 'erytrocyten', 'albumineratio', 'aspect supernatant',
                        'immuno-elektroforese liquor na concentra', 'osmolaliteit',
                        'huid        - verwijderen retentie cyste', 'ro-gel. als ass. bij inbrengen endoproth',
                        'lymf.syst.  - regionale lymfeklierdissec', 'brachytherapie - interstitieel - bijzond',
                        'echo buikwand', 'lever       - excisie tumor of metastase',
                        'lymf.syst.  - sentinel node procedure li', 'bekken',
                        'niet declarabele dagverpleging-bv klin.p', 'hele bovenbeen met onderbeen en voet',
                        'vrw.gesl.org- second-look-oper.ovariumca', 'wisselen drain onder rontgencontrole',
                        'doorlichting bij niersteenvergruizing', 'verwisselen van nefrostomie drain',
                        'oesofagografie - oraal', 'rontgenonderzoek onvolledig', 'knie - m-z onderbeen',
                        'mri art. en ven.abdomen', 'ct art. en ven.pulmonales', 'mec-a-bepaling voor mrsa mbv pcr',
                        'vagina      - extirpatie van vagina met', 'ureter - splintografie cq opspuiten uret',
                        'neuronspecifiek enolase', 'ct knie', 'interc.consult poliklinisch', 'echo thorax', 'obductie',
                        'gentamycine kwn - immuno-assay', 'anesthesie  - epiduraal met bloedpatch',
                        'dikke darm  - resectie sigmoid met prima', 'dikke darm  - appendectomie acuut via me',
                        'totaal t3 - trijoodthyronine mbv radiois', 't3-uptake latente bindingscap. geb. tbg',
                        'echo bekken inhoud muv anatomiecode 83 t', 'uterus      - tot. laparoscopische hyste',
                        'wervelkolom - cervicaal', 'wervelkolom - thoracaal', 'wervelkolom - thoracolumbale overgang',
                        'plaatsen van v.cava filter', 'interc.consult klinisch  urologie',
                        'microscopisch onderzoek  - elektronenmic', 'eerste hulp - niet seh afdeling - elders',
                        'buik        - second-look-oper.ovariumca', 'itraconazol mbv hplc',
                        'bepaling hepatitis-be-antigeen', 'bepaling antist. hepatitis-be-antigeen']

    seq_features = ['Producer code', 'Number of executions', 'org:group', 'Section']

    int2act = dict(zip(range(len(seq_act_features)), seq_act_features))

    # seq_features = seq_act_features + seq_features

    df = pd.read_csv(ds_path)

    # order instances by timestamp of first event
    mapping = []
    for case in df['Case ID'].unique():
        df_tmp = df[df['Case ID'] == case]
        df_tmp = df_tmp.to_numpy()
        mapping.append([df_tmp[0, 0], df_tmp[0, 2]])
    mapping = pd.DataFrame(mapping, columns=['id', 'time'])
    mapping['time'] = pd.to_datetime(mapping['time'])
    mapping = mapping.sort_values(by='time')
    mapping = dict(zip(mapping['id'], np.arange(len(mapping['id']))))
    df['Case ID'].replace(mapping, inplace=True)
    df = df.sort_values(['Case ID', 'Complete Timestamp'])
    df = df.reset_index(drop=True)

    # filter out columns
    not_relevant_cols = ['Start Timestamp', 'Variant', 'Variant index',
                         'Activity code', 'Age:1', 'Age:2', 'Age:3']
    df.drop(not_relevant_cols, axis=1, inplace=True)

    null_cols = df.columns[df.isnull().sum() / len(df) > 0.99]
    df.drop(null_cols, axis=1, inplace=True)
    for col in df.columns:
        if len(df[col].unique()) == 1:
            df.drop(col, inplace=True, axis=1)

    # time feature
    df['Complete Timestamp'] = pd.to_datetime(df['Complete Timestamp'])
    cat_features = ['Treatment code', 'Diagnosis code',
                    'Treatment code:3', 'Treatment code:2', 'Treatment code:1',
                    'Specialism code:3', 'Specialism code:2', 'Specialism code:1',
                    'Diagnosis code:3', 'Diagnosis code:2', 'Diagnosis code:1',
                    'Diagnosis Treatment Combination ID', 'Diagnosis Treatment Combination ID:3',
                    'Diagnosis Treatment Combination ID:2', 'Diagnosis Treatment Combination ID:1',
                    'Diagnosis', 'Diagnosis:3', 'Diagnosis:2', 'Diagnosis:1',
                    'Producer code', 'Number of executions', 'org:group', 'Section', 'Diagnosis:4',
                    'Treatment code:4', 'Specialism code:4', 'Diagnosis code:4',
                    'Diagnosis Treatment Combination ID:4']

    # cat features
    for cat_feature in cat_features:
        df[cat_feature] = df[cat_feature].fillna(-1)
        mapping = dict(zip(df[cat_feature].unique(), np.arange(len(df[cat_feature].unique()))))  # ordinal encoding
        df[cat_feature] = df[cat_feature].apply(lambda x: mapping[x])
        max_ = max(df[cat_feature])
        df[cat_feature] = df[cat_feature].apply(lambda x: x / max_)  # normalise ordinal encoding

    # num features
    df['Age'] = df['Age'].fillna(-1)
    _max = max(df['Age'])
    df['Age'] = df['Age'].apply(lambda x: x / _max)

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
            if idx == 0:
                x_statics.append(x[static_features].values.astype(float))
                x_time_vals.append([])
                x_seqs.append([])
                after_registration_flag = True

            if x['Activity'] == target_activity and after_registration_flag:
                # if "spoed" in x['Activity'] and after_registration_flag:
                found_target_flag = True

            if after_registration_flag:

                seq_features_vals = []
                for seq_feature_ in seq_features:
                    seq_features_vals.append(x[seq_feature_])

                x_seqs[-1].append(np.array(list(util.get_one_hot_of_activity_bpi2011(x, int2act)) + seq_features_vals))
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

    return x_seqs_, x_statics_, y_, x_time_vals_, seq_features, static_features
