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

    static_features = ['ethnicity', 'gender', 'language', 'religion']

    seq_bin_features = ['diagnosis_NEWBORN', 'diagnosis_PNEUMONIA', 'diagnosis_SEPSIS',
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

    seq_features = ['admission_type', 'insurance', 'marital_status', 'age', 'age_dead']

    int2act = dict(zip(range(len(seq_act_features)), seq_act_features))

    # static_features = static_features + static_bin_features
    seq_features = seq_features + seq_act_features

    # pre-processing
    df = pd.read_csv(ds_path)

    # remove irrelevant data
    remove_cols = ['dob', 'dod', 'dod_hosp']
    remove_cols = remove_cols + seq_bin_features
    df = df.drop(columns=remove_cols)

    # time feature
    df['Complete Timestamp'] = pd.to_datetime(df['Complete Timestamp'])
    cat_features = ['admission_type', 'insurance', 'language', 'marital_status', 'religion', 'ethnicity', 'gender']

    # cat features
    for cat_feature in cat_features:
        mapping = dict(zip(df[cat_feature].unique(), np.arange(len(df[cat_feature].unique()))))  # ordinal encoding
        df[cat_feature] = df[cat_feature].apply(lambda x: mapping[x])
        max_ = max(df[cat_feature])
        df[cat_feature] = df[cat_feature].apply(lambda x: x / max_)  # normalise ordinal encoding

    # num features
    df['age'] = df['age'].fillna(-1)
    df['age_dead'] = df['age_dead'].fillna(-1)
    _max = max(df['age'])
    df['age'] = df['age'].apply(lambda x: x / _max)
    _max = max(df['age_dead'])
    df['age_dead'] = df['age_dead'].apply(lambda x: x / _max)

    # bin features
    bin_features = seq_bin_features
    for bin_feature in bin_features:
        df[bin_feature] = df[bin_feature].fillna(0)

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
        x_past = 0
        for _, x in df_tmp.iterrows():
            idx = idx + 1
            if idx == 0:
                x_statics.append(x[static_features].values.astype(float))
                x_time_vals.append([])
                x_seqs.append([])
                after_registration_flag = True

            if x['Activity'] == target_activity and after_registration_flag:
                found_target_flag = True

            if after_registration_flag:
                seq_features_vals = []
                for seq_feature_ in ['admission_type', 'insurance', 'marital_status', 'age', 'age_dead']:

                    # correct values
                    if idx > 0:
                        if seq_feature_ in seq_bin_features:
                            if x_past[seq_feature_] == 1:
                                x[seq_feature_] = 1

                    seq_features_vals.append(x[seq_feature_])

                x_seqs[-1].append(np.array(list(util.get_one_hot_of_activity(x)) + seq_features_vals))
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


def get_bpi11_data(target_activity, max_len, min_len):

    ds_path = '../data/Hospital log_select_cols_seq_len.csv'

    static_features = ['Treatment code', 'Diagnosis code',
                       'Treatment code:3', 'Treatment code:2', 'Treatment code:1',
                       'Specialism code:3', 'Specialism code:2', 'Specialism code:1',
                       'Diagnosis code:3', 'Diagnosis code:2', 'Diagnosis code:1', 'Age',
                       'Diagnosis Treatment Combination ID', 'Diagnosis Treatment Combination ID:3',
                       'Diagnosis Treatment Combination ID:2', 'Diagnosis Treatment Combination ID:1',
                       'Diagnosis', 'Diagnosis:3', 'Diagnosis:2', 'Diagnosis:1']

    seq_act_features = ['1e consult poliklinisch', 'administratief tarief       - eerste pol', 'verlosk.-gynaec. korte kaart kosten-out', 'echografie  - genitalia interna', 'simulator - gebruik voor aanvang megavol', 'behandeltijd - eenheid t3 - megavolt', 'teletherapie - megavolt fotonen bestrali', 'aanname laboratoriumonderzoek', 'ureum', 'hemoglobine foto-elektrisch', 'creatinine', 'natrium vlamfotometrisch', 'kalium potentiometrisch', 'leukocyten tellen elektronisch', 'trombocyten tellen - elektronisch', 'ordertarief', 'ligdagen - alle spec.beh.kinderg.-reval.', 'sediment - spoed', 'bacteriologisch onderzoek met kweek -nie', 'resistentiebepalingen - 5 bepalingen', 'hepatitis-b surface antigeen confirmatie', 'urine onderzoek - kwalitatief', '190021 klinische opname a002', '190205 klasse 3b        a205', 'sgot - asat kinetisch', 'sgpt - alat kinetisch', 'melkzuurdehydrogenase -ldh- kinetisch', 'differentiele telling automatisch', 'trombotest', 'crp c-reactief proteine', 'brachytherapie - interstitieel - intensi', 'inwend.geneesk.  korte kaart kosten-out', 'vervolgconsult poliklinisch', 'inwend.geneesk.   aanv.kaart kosten-out', 'squamous cell carcinoma mbv eia', 'ca-125 mbv meia', 'ct abdomen', 'telefonisch consult', 'patient niet verschenen radiologie', 'homocystine chromatografisch', 'vulva       - rad.vulvectomie - opp.en d', 'histologisch onderzoek - grote resectiep', 'sediment', 'microscopisch onderzoek - gekleurd en on', 'coupe ter inzage', 'bilirubine -geconjugeerd', 'bilirubine - totaal', 'glucose', 'alkalische fosfatase -kinetisch-', 'bloedgroep abo en rhesusfactor', 'rhesusfactor d - centrifugeermethode - e', 'gammaglutamyltranspeptidase', 'cea - tumormarker mbv meia', 'calcium', 'albumine', 'screening antistoffen erytrocyten', 'thorax', 'mri bekken', 'skelet - scintigrafie totale lichaam', 'verlosk.-gynaec.   jaarkaart kosten-out', 'cytologisch onderzoek - ectocervix -', '190101 bovenreg.toesl.  a101', 'vrw.gesl.org- biopsie-punctie-cytologie', 'vrw.gesl.org- curettage gefractioneerd', 'vulva       - biopsie-punctie-cytologie', 'uterus      - hysteroscopie', 'histologisch onderzoek - kleine resectie', 'ligase chain reaction -lcr-', 'dagverpleging - alle spec.beh.kind.-rev.', 'uterus      - intra-uterine device inbre', '190035 dagverpleging    a007', 'cytologisch onderzoek - vagina -', 'mammografie thoraxwand', 'bot - densitometrie lwk', 'bot - densitometrie femurhals', 'lymfeklier - scint sentinel node dynamis', 'lymfeklier - scint sentinel node vervolg', 'lymfeklier - scint sentinel node met pro', 'hematocriet mbv centrifuge', 'vulva       - radicale vulvect. znd ingu', 'vriescoupe', 'verlosk.-gynaec.  aanv.kaart kosten-out', 'fosfaat', 'alfa-1-foetoproteine', 'ct thorax', 'bezinkingssnelheid foto-elektrisch', 'bekken - staand', 'heup', 'mri wervelkolom - lumbaal', 'mri heup', 'magnesium  -aas', 'echo nieren-urinewegen', 'hematologie - ontstek.scint.tot.lichaam', 'haemoglobine foto-electrisch - spoed', 'ovarium     - adnex-extirpatie dmv lapar', '1e consult      bezoek', 'histologisch onderzoek - biopten nno', 'vulva       - vulvectomie zonder lieskli', 'immunopathologisch onderzoek', 'cito histologisch onderzoek', 'cytologisch onderzoek - diversen -', 'tsh enzym-immunologisch', 'vrij t4 mbv ria -ft 4', 'totaal t4 - thyroxine immunofluorimetris', 'e.c.g.      - elektrocardiografie', '190204 klasse 3a        a204', 'vagina      - biopsie-punctie-cytologie', 'blaas       - cystoscopie', 'vagina      - toucher onder anesthesie', 'blaas       - uretrocystoscopie nno', 'leucocyten tellen -electronisch- spoed', 'crp c-reactief proteine - spoed', 'pyelografie --wo niertransplantaat--', 'mri abdomen', 'cefalinetijd - coagulatie', 'protrombinetijd', 'colon inloop', 'ureum - spoed', 'creatinine - spoed', 'alkalische fosfatase - kinetisch - spoed', 'calcium - spoed', 'natrium - vlamfotometrisch - spoed', 'kalium vlamfotometrisch - spoed', 'totaal eiwit colorimetrisch - spoed', 'sgpt alat kinetisch - spoed', 'sgot asat kinetisch - spoed', 'trombocyten tellen - spoed', 'gamma glutamyltranspeptidase - spoed', 'albumine - spoed', 'magnesium  -aas - spoed', 'dieet nno', 'uterus      - extirpatie - abdominaal me', 'cytologisch onderzoek - ascites -', 'glucose - spoed', 'melkzuur enzymatisch - spoed', 'o2-saturatie', 'echo bovenbuik', 'behandeltijd - eenheid t2 - megavolt', 'triglyceriden - enzymatisch', 'vitamine b12 mbv chemieluminescentie', 'onderzoek van elders niet beoordeeld', 'bilirubine kwantitatief totaal of direct', 'urinezuur met uricase u.v. spectrofotome', 'melkzuur dehydrogenase - spoed', 'cholesterol totaal', 'betasubunit bep. mbv. ria hcg', 'cholesterol hdl', 'fsh enzym-immunologisch', 'lh mbv radioisotopen', 'progesteron mbv radioisotopen', 'prolactine mbv eia', 'oestradiol mbv radioisotopen -ria', 'analgesie   - epiduraal door anesthesist', 'vrw.gesl.org- plastische operatie van vu', 'bicarbonaat', 'actuele ph - pco2 - stand.bicarbonaat', 'testosteron mbv radioisotopen', 'sex-hormone-binding globuline mbv ria', 'urinewegen  - urodynamisch onderzoek-vij', 'differentiatie leukocyten - handmatig', 'punctie tbv cytologisch onderzoek door p', 'cytologisch onderzoek - urine niercyste', 'cervix      - lisexcisie portio - diathe', 'vagina      - scopie incl.evt.vulvabiops', 'protrombinetijd -quick owren of modif.-', 'heparinebepaling - kaoline - cefalinetij', 'international normalised ratio mbv tromb', 'vagina      - tension-free vaginal tape', 'vagina      - colpocleisis lefort', 'cde fenotypering', 'bloedgroepantigenen andere dan abo-rhesu', 'lymf.syst.  - stageringslymfadenectomie', 'widal agglutinatie - yersinia 3', 'widal agglutinatie - yersinia 9', 'amylase', 'foliumzuur mbv radioisotopen', 'dikke darm  - colonoscopie met biopsie', 'inwend.geneesk.    jaarkaart kosten-out', 'infuus      - inbrengen', 'hemoglobine a1', 'vulva       - vulvectomie - liesblok', 'ct bekken', 'echo mamma', 'interc.consult klinisch  anesthesie', 'wervelkolom - lumbaal', 'vulva       - ruime lokale excisie van a', 'behandeltijd - eenheid t1 - megavolt', 'mri grote hersenen', 'ct bovenbuik', 'kruisproef volledig -drie methoden-', '190034 afwezigheidsdag  a006', 'gefiltreerd erytrocytenconcentraat', 'interc.consult klinisch  chirurgie', 'alfa amylase - spoed', 'chloride - spoed', 'huidplastiek- z-plastiek huid hoofd en h', 'cytologisch onderzoek - lymfeklierpuncti', 'brain natriuretic peptide -bnp- mbv immu', 'vulva       - partiele vulvectomie', 'methemoglobine - sulfhemoglobine elk', 'co-hb kwn.', 'buikoverzicht', 'klinische kaart - anesthesie', 'uterus      - extirpatie abdominaal tota', 'cervix      - conisatie', 'eiwit -colorimetrisch- totaal', 'toonaudiometrie - audiologisch centrum', 'hoogfrequente audiometrie - audiologisch', 'patienten kontakt audiologie korter dan', 'anesthesie  - bij behandeling met radium', '17-hydroxyprogesteron mbv ria', 'delta-4-androsteendion', 'dhea-sulfaat met extractie mbv ria', 'ca-19.9 tumormarker', 'hematocriet - spoed', 'buik        - stageringslaparotom.-oment', 'vrw.gesl.org- exc.destruct.path.afwijkin', 'echografie  - blaas - extern', 'totaal eiwit', 'vitamine b1 - thiamine', 'bekken - liggend', 'anti-hav ig of igg-elisa of ria- in een', 'antist.tegen hepatitis-b surface antigee', 'hepatitis-b-anti-core igm', 'anti-hepatitis-c-virus mbv elisa', 'fosfaat - spoed', 'echogeleide als ass.bij punctie-biopsie', 'vagina      - operatie vesicovaginale fi', 'vagina      - scopie excl.weefselwegname', 'bovenarm', 'diagnostische punctie onder echocontrole', 'echo onderbuik', 'ct-gel. als ass. bij punctie - biopsie i', 'hyperthermie', 'cytologisch onderzoek - endocervix -', 'oestron mbv radioisotopen', 'vrw.gesl.org- adnex-extirpatie dmv lapar', 'cytologisch onderzoek - leverpunctie -', 'kreatinine fosfokinase - spoed', 'creatine fosfokinase mb - spoed', 'fibrinogeen', 'antitrombine iii mbv chromogeen-n-substr', 'vagina      - vaginotomie', 'ovarium     - redebulking ovarium carcin', 'directe antiglobulinetest - coombs', 'bloedgroepbep. anders dan abo per bloedg', 'antistoffen tegen erys - specificiteit -', 'uitsluiten irregulaire a.s. tegen erys d', 'ident. irregulaire a.s. tegen erys na po', 'bloedgroepbepaling anders dan abo  kidd', 'echografie a vue ivm zwangerschap - rout', 'echografie a vue ivw zwangerschap - met', 'duplexscan - venen been', 'vrw.gesl.org- adhesiolysis ovarium plus', 'chloride', 'klinische kaart - verloskunde en gynaeco', 'ct hals', 'ijzer', 'reticulocyten tellen mbv facscan', 'ferritine kwn mbv chemieluminescentie', 'transferrine mbv ics', 'behandeltijd - eenheid t4 - megavolt', 'mri wervelkolom - thoracaal', 'buik        - stag.lap-tumorred.-omentec', 'ct wervelkolom - lumbaal', 'ovarium     - debulking ovariumcarcinoom', 'echo abdomen', 'zwangerschapsreactie - spoed', 'ct pancreas en milt', 'ca 15.3 - tumormarker mbv meia', 'cytologisch onderzoek tbv bevolkingsonde', 'blaas       - cystoscopie met biopsie', 'onderz. naar aanwezigheid van koude anti', 'uterus      - extirpatie vaginaal totaal', 'nucleair onderzoek van elders', 'melkzuur enzymatisch', 'infuus      - bloed', 'echo schouder - m-z bovenarm', 'creatinefosfokinase -kinetisch-', 'creatine fosfokinase isoenzym ckmb', 'troponine-t mbv elisa', 'bovenbeen', 'peritoneum  - omentectomie', 'buik        - laparoscopie - diagnostisc', 'vrw.gesl.org- operatie van clitoris nno', 'cytologisch onderzoek - buiktumorpunctie', 'dikke darm  - appendectomie nno', 'buik        - adhesiolyse en biopsie dmv', 'schoudergewricht', 'endoretrograde cholangiografie', 'pyelografie antegraad via punctie', 'cystografie retrograad', 'echo hals --waaronder schildklier--', 'uterus      - carc. cervix vlgs. werthei', 'mri tractus urogenitalis', 'longfunctie - spirometrie', 'longfunctie - stroomvolume curve', 'interc.consult klinisch  radiotherapie', 'echo pancreas en milt', 'ro-gel. als ass. bij punctie - biopsie l', 'hemoglobinen -abnormaal incl. hba2', 'erytrocyten tellen - elektronisch', 'doorlichting zonder opname buik', 'ro-gel. als ass. bij inbrengen endoproth', 'lymf.syst.  - regionale lymfeklierdissec', 'echo buikwand', 'lymf.syst.  - sentinel node procedure li', 'ct hersenen', 'niet declarabele dagverpleging-bv klin.p', 'tumor - petscan totale lichaam', 'anti-hiv mbv elisa', 'bekken', 'rontgenonderzoek onvolledig', 'vrw.gesl.org- second-look-oper.ovariumca', 'echo been', 'anesthesie  - epiduraal met bloedpatch', 'paclitaxel', 'uterus      - tot. laparoscopische hyste', 'wervelkolom - cervicaal', 'wervelkolom - thoracaal', 'wervelkolom - thoracolumbale overgang', 'eerste hulp - niet seh afdeling - elders', 'itraconazol mbv hplc', 'bepaling hepatitis-be-antigeen', 'bepaling antist. hepatitis-be-antigeen']

    seq_features = ['Producer code', 'Number of executions', 'org:group', 'Section']

    int2act = dict(zip(range(len(seq_act_features)), seq_act_features))

    seq_features = seq_act_features + seq_features

    df = pd.read_csv(ds_path)

    # filter out columns
    null_cols = df.columns[df.isnull().sum()/len(df) > 0.95]
    df.drop(null_cols, axis=1, inplace=True)
    not_relevant_cols = ['Start Timestamp', 'Variant', 'Variant index',
                         'Activity code', 'Age:1', 'Age:2', 'Age:3',
                         'Diagnosis:4', 'Treatment code:4', 'Specialism code:4', 'Diagnosis code:4',
                         'Diagnosis Treatment Combination ID:4']
    df.drop(not_relevant_cols, axis=1, inplace=True)
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
                       'Producer code', 'Number of executions', 'org:group', 'Section']

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
                found_target_flag = True

            if after_registration_flag:

                seq_features_vals = []
                for seq_feature_ in seq_features:
                    seq_features_vals.append(x[seq_feature_])

                x_seqs[-1].append(np.array(list(util.get_one_hot_of_activity(x)) + seq_features_vals))
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

    print(0)

    return 0
