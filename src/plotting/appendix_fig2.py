# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 17:03:40 2021

@author: ov59opom
"""

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

n_groups = 19
index = np.arange(n_groups)


palet = sns.color_palette("tab10")
matplotlib.rcParams.update({'font.size': 16})


#create data
data1 = [0.984084880636605,0.984084880636605,0.984084880636605,0.984084880636605,0.977886977886978,0.977886977886978,0.976377952755906,0.976377952755906,0.96551724137931,0.96551724137931,0.931034482758621,0.931034482758621,0.9375,0.9375,0.888888888888889,0.888888888888889,1,1,1]
data2 = [0.0159151193633952,0.0159151193633952,0.0159151193633952,0.0159151193633952,0.0221130221130221,0.0221130221130221,0.0236220472440944,0.0236220472440944,0.0344827586206896,0.0344827586206896,0.0689655172413793,0.0689655172413793,0.0625,0.0625,0.111111111111111,0.111111111111111,0,0,0]
width =0.30

fig= plt.figure(figsize=(8,8))

#define chart parameters
xloc = np.arange(n_groups)

plt.xlabel('Size of patient pathway prefix for prediction')
plt.ylabel('Percentage')
#plt.title('Distribution of process instance prefixes of the test set')


#display stacked bar chart
p1 = plt.bar(index, data1, width=width, label='No')
p2 = plt.bar(index, data2, bottom=data1, width=width,  label='Yes')
plt.xticks(index, ('1', '2', '3', '4','5', '6', '7', '8','9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19'))
#plt.legend(loc='lower right', title='Left against medical advice')

plt.tight_layout()
plt.savefig('appendix2.pdf', bbox_inches="tight")
