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


#Calculate transferability - move this to functions later
def get_transfer_score(auc_target, auc_source):
    transfer_score = (auc_source - auc_target)/auc_target * 100
    return transfer_score


# Examples of AUCs for transferability
    
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
