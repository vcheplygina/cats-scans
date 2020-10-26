#%%
from src.similarity.mfe_function import feature_extraction
from src.io.expert_answer_import import expert_answers
from src.io.matrix_processing import row_normalization
from src.io.sim_matrix_csv import write_xlsx

""""Define paths and size of potiential subset. Contains one function to carry out whole experiment"""

# General path for both statistical and expert meta-features

absolute_path_local_data = 'C:/Users/20169385/PycharmProjects/cats-scans/local_data'

# General variables of statistical meta-features

datasets_list = ['chest_xray', 'dtd', 'ISIC2018', 'stl-10', 'pcam']

defined_subset = 'None'        # subset for statistical meta-features

#%% Statistical meta-features separate color channels

define_color_channel = 'blue'

stat_mfe = row_normalization(feature_extraction(datasets=datasets_list, mfe_path = absolute_path_local_data+'/datasets', mfe_subset=defined_subset, color_channel=define_color_channel))

print(stat_mfe)

#%% Statisical meta-features combined color_channels

combined_stat_mfe = row_normalization((((feature_extraction(datasets=datasets_list, mfe_path = absolute_path_local_data+'/datasets', mfe_subset=defined_subset, color_channel='blue'))
                                   + (feature_extraction(datasets=datasets_list, mfe_path = absolute_path_local_data+'/datasets', mfe_subset=defined_subset, color_channel='green'))
                                   + (feature_extraction(datasets=datasets_list, mfe_path = absolute_path_local_data+'/datasets', mfe_subset=defined_subset, color_channel='red')))/3))

print(combined_stat_mfe)

#%% Expert input meta-features

expert_mfe = row_normalization(expert_answers(expert_answer_path=absolute_path_local_data))

print(expert_mfe)

#%% If one wants the results to be written in an excel file please fill in the following arguments

excel_file_path = 'C:/Users/20169385/Desktop/Universiteit/Courses/Year 4/Q1/BEP/'

meta_feature_type = combined_stat_mfe     #stat_mfe or expert_mfe

excel_file_name = 'stats_combined_mfe'

write_xlsx(xlsx_path=excel_file_path, xlsx_name=excel_file_name+'.xlsx', data_list=datasets_list, sim_mat=meta_feature_type)
