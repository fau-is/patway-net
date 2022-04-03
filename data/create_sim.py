import pandas as pd
import random
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

# Define number and length of cases
num_cases = 5000  # 50000
length = 12
num_events = num_cases * length

# Variables for timestamp
time_str = '01/01/2020 08:00:00'
date_format_str = '%d/%m/%Y %H:%M:%S'
time = datetime.strptime(time_str, date_format_str)

# Create lists for further use
label_list_short = []
label_list_cases = []
class_list_short = []
class_list_cases = []
foreigner_list = []
age_list = []
bmi_list = []
gender_list = []
crp_list = []
lacticacid_list = []
pattern_list = []

# Basic structure of data frame
df_main = pd.DataFrame(
    columns=['Case ID', 'Activity', 'Timestamp', 'Gender', 'Foreigner', 'Age', 'BMI', 'CRP', 'LacticAcid', 'Label'])

# Create one instance
for idx in range(0, num_cases):

    print(idx)

    # Case ID
    caseid = [idx] * length

    # Activity
    acts = ['CRP', 'CRP', 'CRP', 'CRP', 'CRP', 'LacticAcid', 'LacticAcid', 'LacticAcid', 'IVA', 'IVA', 'IVL']
    random.shuffle(acts)
    acts = ['Start'] + acts

    # Timestamp
    tmstmp = []
    # Create datetime object from timestamp string
    n = 1

    for i in range(0, length):
        # Add 1 minute to datetime object
        time = time + pd.DateOffset(minutes=n)
        tmstmp.append(time)

    time = time + pd.DateOffset(days=n) - length * pd.DateOffset(minutes=n)

    # Gender (male)
    gender_rnd = random.randint(0, 1)
    gender = [gender_rnd] * length

    # Foreigner
    foreigner_rnd = random.randint(0, 1)
    foreigner = [foreigner_rnd] * length

    # Age (normalized to values between 0 and 1)
    age_rnd = random.randrange(0, 100, 1) / 100
    age = [age_rnd] * length

    # BMI
    bmi_rnd = random.randrange(15, 50, 1)
    bmi = [bmi_rnd] * length

    # CRP with r as factor
    r = 0.03
    crp = []
    rnd = random.randrange(5, 580, 1)

    # Determine whether do increase or decrease initial value with random number; a value of 1 implies an increase, while a value of 0 implies a decrease
    crp_rnd = random.randint(0, 1)

    for i in range(0, 5):
        if crp_rnd == 1:
            rnd = rnd * (1 + r)
        else:
            rnd = rnd / (1 + r)
        crp.append(rnd)

    # Place CRP values to correspondent activities
    crp_temp = [0] * length
    crp_final = []

    for i in range(0, length):
        if acts[i] == 'CRP':
            crp_temp[i] = 1

    idx = 0

    for i in range(0, length):
        if crp_temp[i] == 1:
            crp_final.append(crp[idx])
            idx = idx + 1
        else:
            crp_final.append(np.nan)

    # LacticAcid
    lacticacid = [random.randrange(2, 45, 1) / 10.0 for p in range(0, 3)]

    # Place LacticAcid values to correspondent activities 
    lacticacid_temp = [0] * length
    lacticacid_final = []

    for i in range(0, length):
        if acts[i] == 'LacticAcid':
            lacticacid_temp[i] = 1

    idx = 0

    for i in range(0, length):
        if lacticacid_temp[i] == 1:
            lacticacid_final.append(lacticacid[idx])
            idx = idx + 1
        else:
            lacticacid_final.append(np.nan)

    # Check if certain patterns occur in instance    
    pattern1 = ['IVA', 'IVA', 'IVL']
    pattern2 = ['IVL', 'IVA', 'IVA']


    def contains_sublist(l1, l2):
        index_list = [i for i, v in enumerate(l1) if v == l2[0]]
        for ii in index_list:
            l1_slice = l1[ii:ii + len(l2)]
            if l1_slice == l2:
                return True
        else:
            return False


    if contains_sublist(acts, pattern1) or contains_sublist(acts, pattern2) == True:
        pattern_occurance = 1
    else:
        pattern_occurance = 0

    # Create label based on rules 
    weight = 0.25

    label_init = weight * gender_rnd + (-(age_rnd - 0.5) ** 2 + weight) + weight * crp_rnd + weight * pattern_occurance
    label = [label_init] * length

    # Concatenate vectors and transpose matrix
    concat_case = np.vstack(
        (caseid, acts, tmstmp, gender, foreigner, age, bmi, crp_final, lacticacid_final, label)).transpose()

    # Create case
    df_temp = pd.DataFrame(data=concat_case,
                           columns=['Case ID', 'Activity', 'Timestamp', 'Gender', 'Foreigner', 'Age', 'BMI', 'CRP',
                                    'LacticAcid', 'Label'])

    df_main = df_main.append(df_temp, ignore_index=True)

    label_list_short.append(label_init)
    label_list_cases.extend(label)

    gender_list.append(gender_rnd)
    foreigner_list.append(foreigner_rnd)
    age_list.append(age_rnd)
    bmi_list.append(bmi_rnd)
    crp_list.append(crp_rnd)

    pattern_list.append(pattern_occurance)

# Create label for classes
threshold = np.median(label_list_short)
class_rea_final = []

for i in range(0, num_events):
    if label_list_cases[i] > threshold:
        class_rea_final.append(1)

    else:
        class_rea_final.append(0)

# Add column with class labels to data frame
# class_rea_final = class_rea_final.transpose()
df_main.insert(loc=10, column='Release A', value=class_rea_final)

# Save data frame as csv
df_main.to_csv(f'Simulation_data_{num_cases}.csv')

"""
#####################
### VISUALIZATION ###

# Take every 12th value from class list to get a short list for visualization
class_rea_short = class_rea_final[::length]

fig_data = np.vstack((gender_list, foreigner_list, age_list, bmi_list, crp_list, pattern_list, label_list_short)).transpose()

df_fig = pd.DataFrame(data=fig_data, columns=['Gender', 'Foreigner', 'Age', 'BMI', 'CRP', 'Activity Pattern', 'Label'])

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
 
fig.suptitle('Labels')
 
 
sns.scatterplot(ax=axes[0, 0], data=df_fig, x='Gender', y='Label')
sns.scatterplot(ax=axes[0, 1], data=df_fig, x='Foreigner', y='Label')
sns.scatterplot(ax=axes[0, 2], data=df_fig, x='Age', y='Label')
sns.scatterplot(ax=axes[1, 0], data=df_fig, x='BMI', y='Label')
sns.scatterplot(ax=axes[1, 1], data=df_fig, x='CRP', y='Label')
sns.scatterplot(ax=axes[1, 2], data=df_fig, x='Activity Pattern', y='Label')




#######################
### FURTHER METRICS ###

# Show density of labels
#sns.set_style('whitegrid')
#sns.kdeplot(np.array(label_list_short), bw_method=0.5)

# Print some metrics for check
#print('Weight:', weight)
#print('Threshold:', threshold)

#print('Mean of labels:', np.mean(label_list_short))
#print('Median of labels:', np.median(label_list_short))

#print('Mean of classes:', np.mean(class_rea_final))

"""
