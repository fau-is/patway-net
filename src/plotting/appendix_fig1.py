# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 14:34:32 2021

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

data1 = [1484,1484,1484,1484,398,398,124,124,56,56,27,27,15,15,8,8,6,6,5]
data2 = [24,24,24,24,9,9,3,3,2,2,2,2,1,1,1,1,0,0,0]
width =0.30

fig= plt.figure(figsize=(8,8))

plt.xlabel('Size of patient pathway prefix for prediction')
plt.ylabel('Amount')
#plt.title('Amount of labels per process instance prefixes of the test set')
plt.bar(np.arange(len(data1)), data1, width=width, label='No')
plt.bar(np.arange(len(data2))+ width, data2, width=width, label='Yes')
plt.xticks(index + 0.5*width, ('1', '2', '3', '4','5', '6', '7', '8','9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19'))
plt.legend(title='Left against medical advice')

plt.tight_layout()
plt.savefig('appendix1.pdf', bbox_inches="tight")
