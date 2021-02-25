# Import packages
import numpy as np
from src.similarity.mfe_function import feature_extraction
from src.io.expert_answer_import import expert_answers
from src.io.matrix_processing import norm_and_invert
from src.evaluation.numpy_to_heatmap import make_heatmap
from src.evaluation.make_bar_chart import make_bars

# Below the absolute path to the local_data folder (which should be downloaded) should be specified

absolute_path_local_data = 'C:/Users/20169385/PycharmProjects/cats-scans/local_data'

# Specify a list containing all the names of the datasets you want to compare (case sensitive)

datasets_list = ['chest_xray', 'dtd', 'ISIC2018', 'stl-10', 'pcam']     # Possible dataset names: 'chest_xray', 'dtd', 'ISIC2018', 'stl-10' and 'pcam'

# Specify size of subset. Should be above 50 to avoid errors with regard to number of labels.

defined_subset = 'None'     # Besides numerical values, 'None' is also an option. In this case a maximum of 15.000 images of every dataset is taken to avoid a memory error

# Specify location were figure will be saved

save_path = 'C:/Users/20169385/PycharmProjects/cats-scans/outputs'

#%% Calculate statistical similarity matrix for all color channels combined

save_name = 'stat_heatmap'  # Define the name of the output figure

stat_sim = norm_and_invert((((feature_extraction(datasets=datasets_list, mfe_path = absolute_path_local_data+'/datasets', mfe_subset=defined_subset, color_channel='blue'))
                                   + (feature_extraction(datasets=datasets_list, mfe_path = absolute_path_local_data+'/datasets', mfe_subset=defined_subset, color_channel='green'))
                                   + (feature_extraction(datasets=datasets_list, mfe_path = absolute_path_local_data+'/datasets', mfe_subset=defined_subset, color_channel='red')))/3))

stat_heatmap = make_heatmap(stat_sim, data_list = datasets_list, name = save_name, output_path = save_path)

#%% Calculate experts similarity matrix

save_name = 'exp_heatmap'   # Define the name of the output figure

expert_sim = norm_and_invert(expert_answers(expert_answer_path=absolute_path_local_data))

make_heatmap(expert_sim, data_list = datasets_list, name = save_name, output_path = save_path)

#%% Load in AUC scores

save_name = 'auc_heatmap'   # Define the name of the output figure

auc_scores = np.load(absolute_path_local_data + '/target_auc.npy')

auc_heatmap = make_heatmap(auc_scores, data_list = datasets_list, name = save_name, output_path = save_path, auc='yes')

#%% Make bar charts (every section has to have been executed before this section can run)

target_data = 'chest_xray'    # Choose target dataset (options: 'chest_xray', 'ISIC2018', 'pcam')

make_bars(datalist = datasets_list, target_dataset = target_data,
          auc_mat = auc_scores, stat_mat = stat_sim, exp_mat = expert_sim, name = target_data + '_bars', output_path = save_path)
