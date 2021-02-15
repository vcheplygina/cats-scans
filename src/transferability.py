# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 09:43:02 2021

@author: vcheplyg
"""

# Import packages
import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#Calculate transferability - move this to functions later
def get_transfer_score(auc_target, auc_source):
    transfer_score = (auc_source - auc_target)/auc_target * 100
    return transfer_score




# Examples of AUCs for transferability
def test_transfer_score():    
    #Improves somewhat
    auc_target_only1 = 0.7
    auc_with_source1 = 0.8
    print(get_transfer_score(auc_target_only1, auc_with_source1))
    
    
    #Improves an already easy case
    auc_target_only2 = 0.9
    auc_with_source2 = 0.95
    print(get_transfer_score(auc_target_only2, auc_with_source2))
    
    
    #Can't learn without it
    auc_target_only3 = 0.5
    auc_with_source3 = 1
    print(get_transfer_score(auc_target_only3, auc_with_source3))
    
    
    #Makes things worse
    auc_target_only4 = 0.7
    auc_with_source4 = 0.6
    print(get_transfer_score(auc_target_only4, auc_with_source4))


# Create transferability plot like in the paper "Geometric Dataset Distances via Optimal Transport"
def plot_distance_score(distance, score_mean, score_std, labels):
    
    # Plot the data
    plt.errorbar(distance, score_mean, yerr=score_std, fmt='o')

    # Label points    
    offset = 0.2
    
    for i, label in enumerate(labels):
        plt.annotate(label, (distance[i]+offset, score_mean[i]+offset))
        #u'\u03b8 (°)', fontdict={'fontname': 'Times New Roman'}
    
    
    # Regression line
    sns.regplot(distance, score_mean)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(distance, score_mean)
    
    str = "rho={:1.2f}".format(r_value)
    plt.legend([str])
    
    
    # Plot details
    
    plt.xlabel("Dataset distance")
    plt.ylabel("Relative AUC increase")
    
    # Save plot (TODO)
    

score_mean = [7, 9, 11, 14, 13, 16]
score_std = [1, 1, 1, 1, 1, 1]

dataset_distance = [5, 10, 15, 20, 25, 30]

labels = ["A to B", "A to C", "A to D", "B to C", "B to D", "C to D"]

plot_distance_score(dataset_distance, score_mean, score_std, labels)

