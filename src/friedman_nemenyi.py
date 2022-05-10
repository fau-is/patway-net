# -*- coding: utf-8 -*-
"""
Created on Mon May  9 16:08:06 2022

@author: ov59opom
"""


# Importing libraries
from scipy import stats
import scikit_posthocs as sp
import numpy as np

value_lr = []
value_nb = []
value_knn = []
value_dt = []

# Read and convert each data input
# Logistic Regression
data_lr = np.loadtxt('output/sepsis_lr_Admission IC.txt', delimiter='\t', skiprows=0, dtype=str)
data_lr = data_lr[1:11]

for i in range (0,10) :
    AUC = float((data_lr[i][2:]))
    value_lr.append(AUC)
  
# Naive Bayes
data_nb = np.loadtxt('output/sepsis_nb_Admission IC.txt', delimiter='\t', skiprows=0, dtype=str)
data_nb = data_nb[1:11]

for i in range (0,10) :
    AUC = float((data_nb[i][2:]))
    value_nb.append(AUC)

# kNN
data_knn = np.loadtxt('output/sepsis_knn_Admission IC.txt', delimiter='\t', skiprows=0, dtype=str)
data_knn = data_knn[1:11]

for i in range (0,10) :
    AUC = float((data_knn[i][2:]))
    value_knn.append(AUC)

# Decision Tree 
data_dt = np.loadtxt('output/sepsis_dt_Admission IC.txt', delimiter='\t', skiprows=0, dtype=str)
data_dt = data_dt[1:11]

for i in range (0,10) :
    AUC = float((data_dt[i][2:]))
    value_dt.append(AUC)


#############################

# Conduct the Friedman Test
stats.friedmanchisquare(value_lr, value_nb, value_knn, value_dt)
 
# Combine three groups into one array
data = np.array([value_lr, value_nb, value_knn, value_dt])
 
# Conduct the Nemenyi post-hoc test
#sp.posthoc_nemenyi_friedman(data.T)

print(sp.posthoc_nemenyi_friedman(data.T))

# Explanation
# 0 = Logistic Regression
# 1 = Naive Bayes
# 2 = K-Nearest Neighbor
# 3 = Decision Tree