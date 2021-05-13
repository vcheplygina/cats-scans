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
from scipy import stats, spatial
import pickle

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
    
    plt.figure(figsize=(8,6))
    #plt.ylim([-10,20])
    #plt.xlim([-0.05, 1])
    # Plot the data
    plt.errorbar(distance, score_mean, yerr=score_std, fmt='o')

    # Label points    
    offset = 0.02
    
    for i, label in enumerate(labels):
        plt.annotate(label, (distance[i]+offset, score_mean[i]+offset), size=8)
    
    
    
    # Regression line
    sns.regplot(distance, score_mean)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(distance, score_mean)
    
    str = "rho={:1.2f}, p={:1.2f}".format(r_value, p_value)
    plt.legend([str])
    
    
    # Plot details
    
    plt.xlabel("Dataset distance")
    plt.ylabel("Relative AUC increase")
    plt.tight_layout()
    plt.savefig('transfer_score.png')
    
    
    
    
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


#  Add columns for transfer scores and distances
num_folds = 5
for fold in np.arange(0,num_folds)+1:
    
    col = 'score_'+str(fold)
    aucs[col] = np.nan
    
    col = 'distance_'+str(fold)
    aucs[col] = np.nan
    
    
    
#Calculate transferability
for index, row in aucs.iterrows():
    
    target = row['target']

    #Calculate transfer score
    num_folds = 5
    
    for fold in np.arange(0,num_folds)+1:
        
         baseline = aucs.loc[(aucs['target']==target) & (aucs['source']==target)]
    
         col = str(fold)
    
         target_only = baseline['fold_'+col]
         with_transfer = row['fold_'+col]
         
         score = get_transfer_score(target_only, with_transfer)
    
     
         aucs.at[index,'score_'+col] = score
    
aucs.to_csv('aucs_with_scores.csv')


########################### Transferability vs task2vec distance


path_emb = '/Users/vech/Sync/30-ResearchPapers/cats-scans/cats-scans/results/Task2Vec_embeddings/'
subset_index = np.random.randint(1,100, 5)

os.chdir('/Users/vech/Sync/30-ResearchPapers/cats-scans/cats-scans/') #magic required to load pickles

# Calculate distances
for index, row in aucs.iterrows():
    
    target = row['target']
    source = row['source']
   
    
    if source == 'imagenet':
        source = 'stl10'
   
    
    #Calculate distance
    num_folds = 5
    
    
    for fold in np.arange(0,num_folds)+1:
        col = str(fold)
        subset = str(subset_index[fold-1])
        
        path_target = path_emb+'embedding_' + target + '_subset' + subset + '.p'
        path_source = path_emb+'embedding_' + source + '_subset' + subset + '.p'
 
        
        emb_target = pickle.load(open(path_target, 'rb'))
        emb_source = pickle.load(open(path_source, 'rb'))
        
        dist = spatial.distance.cosine(emb_target.hessian, emb_source.hessian)
        aucs.at[index,'distance_'+col] = dist
        
           
aucs.to_csv('aucs_with_distances.csv')


# Make plot?
distance_col = [col for col in aucs if col.startswith('distance')]
score_col = [col for col in aucs if col.startswith('score')]

labels = aucs['source']+' to ' + aucs['target']


dist = aucs[distance_col].mean(axis=1).to_numpy()
meanauc = aucs[score_col].mean(axis=1).to_numpy()
stdauc = aucs[score_col].std(axis=1).to_numpy()


plot_distance_score(dist, meanauc, stdauc, labels)




########################### Transferability vs expert distance

# Load expert distances
expdist = pd.read_csv('/Users/vech/Sync/30-ResearchPapers/cats-scans/cats-scans/results/experts_clean.csv', sep=";")

# Initialize column in same dataframe
aucs['expert_distance'] = np.nan

# Lookup expert distances from CSV
for index, row in aucs.iterrows():
    
    target = row['target']
    source = row['source']
   
    
    dist = expdist.loc[(expdist['source']==source) & (expdist['target']==target)]
    aucs.at[index, 'expert_distance'] = dist['distance']


dist = aucs['expert_distance'].to_numpy()


plot_distance_score(dist, meanauc, stdauc, labels)