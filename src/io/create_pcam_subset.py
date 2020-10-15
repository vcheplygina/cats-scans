#%%
import pandas as pd
from shutil import copyfile

csv_file = pd.read_csv('C:/Users/20169385/PycharmProjects/cats-scans/local_data/datasets/pcam/PCAM_subset.csv')

for row in range(len(csv_file.index)):
    last_index = csv_file.values[row][1].rfind('/')
    file_name = csv_file.values[row][1][last_index+1:]
    if csv_file.values[row][2] == 0:
        copyfile('C:/Users/20169385/PycharmProjects/cats-scans/local_data/datasets/pcam/pcam_full/' + file_name,
                 'C:/Users/20169385/PycharmProjects/cats-scans/local_data/datasets/pcam/pcam_subset/pcam_label=0/' + file_name)
    elif csv_file.values[row][2] == 1:
        copyfile('C:/Users/20169385/PycharmProjects/cats-scans/local_data/datasets/pcam/pcam_full/' + file_name,
                 'C:/Users/20169385/PycharmProjects/cats-scans/local_data/datasets/pcam/pcam_subset/pcam_label=1/' + file_name)

#%%

# from shutil import copyfile
#
# test_list = ['banded_0034.jpg', 'bubbly_0061.jpg']
#
# for i in test_list:
#     copyfile('C:/Users/20169385/PycharmProjects/cats-scans/local_data/datasets/pcam/folder/'+i, 'C:/Users/20169385/PycharmProjects/cats-scans/local_data/datasets/pcam/images/'+i)


#%%

# test_list = ['banded_0034.jpg', 'bubbly_0061.jpg']
#
# from zipfile import ZipFile
#
# # Create a ZipFile Object and load sample.zip in it
# with ZipFile('C:/Users/20169385/PycharmProjects/cats-scans/local_data/datasets/pcam/pcam.zip', 'r') as zipObj:
#    # Get a list of all archived file names from the zip
#    listOfFileNames = zipObj.namelist()
#    # Iterate over the file names
#    for fileName in listOfFileNames:
#        # Check filename endswith csv
#        if fileName in subset_filenames:
#            # Extract a single file from zip
#            zipObj.extract(fileName, 'C:/Users/20169385/PycharmProjects/cats-scans/local_data/datasets/pcam/images')

# column = 1
# row = 3
#
# last_index = rows.values[row][column].rfind('/')
# print(last_index)
# file_name = rows.values[row][column][last_index+1:]
# print(file_name)



# with open('C:/Users/20169385/PycharmProjects/cats-scans/local_data/datasets/pcam/PCAM_subset.csv') as csvfile:
#     readCSV = csv.reader(csvfile, delimiter=',')
#     for row in readCSV:
#         print(row)
#         # print(row[1])


# with ZipFile('C:/Users/20169385/PycharmProjects/cats-scans/local_data/datasets/pcam/images/DTD.zip', 'r') as zipObj:
#    # Get list of files names in zip
#    listOfiles = zipObj.namelist()
#
# print(listOfiles)
