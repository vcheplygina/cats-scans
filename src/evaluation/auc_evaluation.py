from sklearn.metrics import roc_auc_score
from numpy.random import seed
import tensorflow as tf

seed(1)
tf.random.set_seed(2)


def calculate_AUC(dataset, generator, predictions):
    """
    :param dataset: dataset that is fed to generator
    :param generator: generator feeding images and labels to model
    :param predictions: predictions made on the validation set using the trained model
    :return: One-Vs-Rest AUC for multiclass case, 'normal' AUC for binary case
    """
    # compute OneVsRest multi-class weighted AUC
    if (dataset == 'stl10') | (dataset == 'sti10'):
        # STL10 and STI10 are given to .flow() generators, so labels are stored in y-object of generator
        OneVsRest_auc = roc_auc_score(generator.y, predictions, multi_class='ovr', average='weighted')
    if (dataset == "isic") | (dataset == 'textures') | (dataset == 'kimia'):
        # .flow_from_dataframe() is used to get data, labels can be collected by calling classes object of generator
        OneVsRest_auc = roc_auc_score(generator.classes, predictions, multi_class='ovr', average='weighted')
    # in binary case, compute 'normal' AUC
    elif (dataset == 'chest') | (dataset == 'pcam-small') | (dataset == 'pcam-middle'):
        # .flow_from_dataframe() is used to get data, labels can be collected by calling classes object of generator
        OneVsRest_auc = roc_auc_score(generator.classes, predictions)
    print(f'Validation auc: {OneVsRest_auc}')

    return OneVsRest_auc

