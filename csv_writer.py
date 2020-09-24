import csv
import numpy as np
from requests_osf import upload_to_osf


def create_metrics_csv(model_choice, acc_per_fold, loss_per_fold, auc_per_fold):
    """
    :param model_choice: model architecture to use for convolutional base (i.e. resnet or efficientnet)
    :param acc_per_fold: list containing accuracy scores per fold
    :param loss_per_fold: list containing losses per fold
    :param auc_per_fold: list containing multi-class auc scores per fold
    :return: csv containing metrics per fold and average and std for all folds
    """
    file = open(f'/Users/IrmavandenBrandt/PycharmProjects/cats-scans/model_results/metrics_{model_choice}.csv',
                'w')  # initialize file

    columns = ['run', 'val-accuracy', 'val-loss', 'val-multi-class AUC']  # define columns

    no_folds = len(acc_per_fold)  # obtain number of folds that have been used in the model
    data = [{'run': f'fold {i}', 'accuracy': acc_per_fold[i], 'loss': loss_per_fold[i],
             'multi-class AUC': auc_per_fold[i]} for i in range(no_folds)]  # append data with metrics per fold
    data.extend([{'run': 'average', 'accuracy': np.mean(acc_per_fold), 'loss': np.mean(loss_per_fold),
                  'multi-class AUC': np.mean(auc_per_fold)},
                 {'run': 'std', 'accuracy': np.std(acc_per_fold), 'loss': np.std(loss_per_fold),
                  'multi-class AUC': np.std(auc_per_fold)}])  # append data with average and std for all folds

    writer = csv.DictWriter(file, columns)  # add columns to file
    writer.writeheader()

    # add the dictionary with data to the csv file
    for row in data:
        writer.writerow(row)

    file.close()

    return file



