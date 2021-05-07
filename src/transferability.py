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
    
    str = "rho={:1.2f}, p={:1.2f}".format(r_value, p_value)
    plt.legend([str])
    
    
    # Plot details
    
    plt.xlabel("Dataset distance")
    plt.ylabel("Relative AUC increase")
    
    # Save plot (TODO), increase font size etc. 
    
    
def show_transfer_matrix(scores, labels):
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(scores, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            ax.text(x=j, y=i,s=scores[i, j], va='center', ha='center', size='xx-large')
    
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)


def test_transfer_matrix():
    
    dataset_distance = [5, 10, 15, 20, 25, 30]
    
    score_mean = [7, 9, 13, 14, 11, 16]
    score_std = [1, 1, 1, 1, 1, 1]
    
    labels_pairs = ["A to B", "A to C", "A to D", "B to C", "B to D", "C to D"]
    plot_distance_score(dataset_distance, score_mean, score_std, labels_pairs)
    
    labels = ["A", "B", "C", "D"]
    scores = np.matrix([[70, 75, 80, 75], [60, 70, 60, 80], [90, 95, 80, 85], [80, 60, 60, 65]])
    
    show_transfer_matrix(scores, labels)




# Load transfer experiment AUCs
aucs = pd.read_csv('/Users/vech/Sync/30-ResearchPapers/cats-scans/cats-scans/results/auc_folds_means_std.csv')
aucs = aucs.dropna(axis=0)


#  Add columns for transfer scores
num_folds = 5
for fold in np.arange(0,num_folds)+1:
    
    col = 'score_'+str(fold)
    aucs[col] = np.nan
    
    

for index, row in aucs.iterrows():
    
    target = row['target']
    print(target)
   
    
    #Calculate transfer score
    num_folds = 5
    
    for fold in np.arange(0,num_folds)+1:
        
         baseline = aucs.loc[(aucs['target']==target) & (aucs['source']==target)]
    
         col = str(fold)
    
         without_transfer = baseline['fold_'+col]
         with_transfer = row['fold_'+col]
         
         score = get_transfer_score(with_transfer, without_transfer)
    
     
         aucs.at[index,'score_'+col] = score
    
aucs.to_csv('aucs_with_scores.csv')


#experts = '/Users/vech/Sync/30-ResearchPapers/cats-scans/experts_clean.xlsx'
#experts = pd.read_excel(experts)


emb = '/Users/vech/Sync/30-ResearchPapers/cats-scans/cats-scans/results/task2vec_embeddings.csv'
emb = pd.read_csv(emb)


