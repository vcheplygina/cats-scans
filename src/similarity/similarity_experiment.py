#%%
from src.similarity.mfe_function import feature_extraction

""""Define paths and size of potiential subset. Contains one function to carry out whole experiment"""

datasets_list = ['chest_xray', 'ISIC2018', 'stl-10', 'dtd', 'pcam']

absolute_path_datasets = 'C:/Users/20169385/PycharmProjects/cats-scans/local_data/datasets'

defined_subset = 50

feature_extraction(datasets=datasets_list, mfe_path = absolute_path_datasets, mfe_subset=defined_subset)



