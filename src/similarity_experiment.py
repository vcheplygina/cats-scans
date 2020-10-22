#%%
from src.similarity.mfe_function import feature_extraction
from src.io.expert_answer_import import expert_answers
from src.io.sim_matrix_csv import write_xlsx

""""Define paths and size of potiential subset. Contains one function to carry out whole experiment"""

datasets_list = ['chest_xray', 'dtd', 'ISIC2018', 'stl-10', 'pcam']

absolute_path_local_data = 'C:/Users/20169385/PycharmProjects/cats-scans/local_data'

defined_subset = 250        # subset for statistical meta-features

#%% Statisitcal meta-features

stat_mfe = feature_extraction(datasets=datasets_list, mfe_path = absolute_path_local_data+'/datasets', mfe_subset=defined_subset)
print(stat_mfe)

#%% Expert input meta-features

expert_mfe = expert_answers(expert_answer_path=absolute_path_local_data)
print(expert_mfe)

#%% If one wants the results to be written in an excel file please fill in the following arguments

excel_file_path = 'C:/Users/20169385/Desktop/Universiteit/Courses/Year 4/Q1/BEP/'

excel_file_name = 'similarity_matrix'

write_xlsx(xlsx_path=excel_file_path, xlsx_name=excel_file_name+'.xlsx', data_list=datasets_list, sim_mat=expert_mfe)
