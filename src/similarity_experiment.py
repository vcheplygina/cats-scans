# Import packages
from src.similarity.mfe_function import feature_extraction
from src.io.expert_answer_import import expert_answers
from src.io.matrix_processing import norm_and_invert
from src.io.sim_matrix_excel import write_xlsx

# Below the absolute path to the local_data folder (which should be downloaded) should be specified

absolute_path_local_data = 'C:/Users/20169385/PycharmProjects/cats-scans/local_data'

# Specify a list containing all the names of the datasets you want to compare (case sensitive)

datasets_list = ['chest_xray', 'dtd', 'ISIC2018', 'stl-10', 'pcam']     # Possible dataset names: 'chest_xray', 'dtd', 'ISIC2018', 'stl-10' and 'pcam'

# Specify size of subset. Should be above 50 to avoid errors with regard to number of labels.

defined_subset = 'None'     # Besides numerical values, 'None' is also an option. In this case a maximum of 15.000 images of every dataset is taken to avoid a memory error

#%% Calculate statistical similarity matrix for one single color channel defined below

define_color_channel = 'blue'       # Possible arguments: 'blue', 'green' and 'red'

stat_mfe = norm_and_invert(feature_extraction(datasets=datasets_list, mfe_path = absolute_path_local_data+'/datasets', mfe_subset=defined_subset, color_channel=define_color_channel))

print(stat_mfe)

#%% Calculate statistical similarity matrix for all color channels combined

combined_stat_mfe = norm_and_invert((((feature_extraction(datasets=datasets_list, mfe_path = absolute_path_local_data+'/datasets', mfe_subset=defined_subset, color_channel='blue'))
                                   + (feature_extraction(datasets=datasets_list, mfe_path = absolute_path_local_data+'/datasets', mfe_subset=defined_subset, color_channel='green'))
                                   + (feature_extraction(datasets=datasets_list, mfe_path = absolute_path_local_data+'/datasets', mfe_subset=defined_subset, color_channel='red')))/3))

print(combined_stat_mfe)

#%% Calculate experts similarity matrix

expert_mfe = norm_and_invert(expert_answers(expert_answer_path=absolute_path_local_data))

print(expert_mfe)

#%% If one prefers the matrices to be saved to an excel file please specify variables below and run this section

excel_file_path = 'C:/Users/20169385/Desktop/Universiteit/Courses/Year 4/Q1/BEP/'       # Path were the excel file will be located

meta_feature_type = combined_stat_mfe     # Change variable depending on which matrix you want to save. Possible answers: stat_mfe, combined_stat_mfe and expert_mfe

excel_file_name = 'stats_inv_mfe'       # Specify name of excel file

write_xlsx(xlsx_path=excel_file_path, xlsx_name=excel_file_name+'.xlsx', data_list=datasets_list, sim_mat=meta_feature_type)
