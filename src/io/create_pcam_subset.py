# This code does not have to be executed if you downloaded the local_data folder!

# Import packages
import pandas as pd
from shutil import copyfile

# Reads the file containing all the names of the images that are in the subset

csv_file = pd.read_csv('C:/Users/20169385/PycharmProjects/cats-scans/local_data/datasets/pcam/PCAM_subset.csv')

# Copy all these images to two name folders according to their label

for row in range(len(csv_file.index)):
    last_index = csv_file.values[row][1].rfind('/')
    file_name = csv_file.values[row][1][last_index+1:]
    if csv_file.values[row][2] == 0:
        copyfile('C:/Users/20169385/PycharmProjects/cats-scans/local_data/datasets/pcam/pcam_full/' + file_name,
                 'C:/Users/20169385/PycharmProjects/cats-scans/local_data/datasets/pcam/pcam_subset/pcam_label=0/' + file_name)
    elif csv_file.values[row][2] == 1:
        copyfile('C:/Users/20169385/PycharmProjects/cats-scans/local_data/datasets/pcam/pcam_full/' + file_name,
                 'C:/Users/20169385/PycharmProjects/cats-scans/local_data/datasets/pcam/pcam_subset/pcam_label=1/' + file_name)
