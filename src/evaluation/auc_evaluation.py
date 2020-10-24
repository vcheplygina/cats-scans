from sklearn.metrics import roc_auc_score


def calculate_AUC(target_data, valid_generator, predictions):
    """
    :param target_data: dataset used as target dataset
    :param valid_generator: generator feeding validation images and labels to model
    :param predictions: predictions made on the validation set using the trained model
    :return: One-Vs-Rest AUC for multiclass case, 'normal' AUC for binary case
    """
    # compute OneVsRest multi-class macro AUC on the test set
    if target_data == "isic":
        OneVsRest_auc = roc_auc_score(valid_generator.classes, predictions, multi_class='ovr', average='macro')
    else:
        OneVsRest_auc = roc_auc_score(valid_generator.classes, predictions, average='macro')
    print(f'Validation auc: {OneVsRest_auc}')

    return OneVsRest_auc
