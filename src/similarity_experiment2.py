# Import packages
import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt

from sklearn.manifold import MDS, TSNE

from os.path import dirname, join 
import sys 
sys.path.insert(0, join(dirname(__file__), '..'))

from src.io.matrix_processing import norm_and_invert
from src.io.expert_answer_import import expert_answers
from src.evaluation.numpy_to_heatmap import make_heatmap


#C:\Users\VCheplyg\Dropbox\20-lab\cats-scans\cats-scans

data_path = 'outputs'


#auc = np.load('../outputs/mean_auc_scores.npy')
#auc_dict= pd.read_csv('../outputs/mean_auc_dict.csv')

absolute_path_local_data = 'C:/Users/VCheplyg/Dropbox/20-lab/cats-scans/cats-scans/data'


exp_dist = expert_answers(expert_answer_path=absolute_path_local_data)
#exp_sim = norm_and_invert(exp_dist) - #Not actually needed for embeddings, distance is what we need


datasets_list = ['chest_xray', 'dtd', 'ISIC2018', 'stl-10', 'pcam'] 


############################# MDS

mds = MDS(2, dissimilarity='precomputed', random_state=42, metric=False)
exp_mds = mds.fit_transform(exp_dist)


colors=[1,2,3,4,5]

fig, ax = plt.subplots()

for i in np.arange(len(colors)):
    #ax.scatter(exp_mds[i,0], exp_mds[i,1], c=colors[i], label=colors[i])
    ax.scatter(exp_mds[i,0], exp_mds[i,1], label=datasets_list[i])

ax.legend()
ax.grid(True)
plt.title('MDS')
plt.show()


############################# TSNE

tsne = TSNE(n_components=2, metric='precomputed')
exp_tsne = tsne.fit_transform(exp_dist)


colors=[1,2,3,4,5]

fig, ax = plt.subplots()

for i in np.arange(len(colors)):
    #ax.scatter(exp_mds[i,0], exp_mds[i,1], c=colors[i], label=colors[i])
    ax.scatter(exp_tsne[i,0], exp_tsne[i,1], label=datasets_list[i])

ax.legend()
ax.grid(True)
plt.title('TSNE')
plt.show()