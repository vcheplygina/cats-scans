import pandas as pd
import os
import pickle

# set embedding directory
embeddings_dir = '/Users/IrmavandenBrandt/Downloads/Internship/Task2Vec_embeddings'
# get embedding paths by selecting files from directory that end with .p
embeddings = [os.path.join(embeddings_dir, f) for f in os.listdir(embeddings_dir) if f.endswith('.p')]

dataframe_entries = []  # initiliaze empty list that will store entries for dataframe


def load_embedding(filename):
    with open(filename, 'rb') as f:
        e = pickle.load(f)
    return e


for e, embedding_path in enumerate(embeddings):
    embedding = load_embedding(embedding_path)
    entry = pd.DataFrame([[embedding.hessian]], columns=['embedding'])
    entry['name'] = embedding_path[65:-2]  # add embedding name in dataframe in column 'name'
    dataframe_entries.append(entry)  # combine entry with other entries for dataframe

dataframe = pd.concat(dataframe_entries, ignore_index=True)  # create dataframe from list of tables and reset index

# save dataframe in the results folder
dataframe.to_csv('/Users/IrmavandenBrandt/PycharmProjects/cats-scans/results/task2vec_embeddings.csv')