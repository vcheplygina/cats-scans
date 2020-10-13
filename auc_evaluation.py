# %%
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from data_import import import_ISIC
import numpy as np
from tf_generators_models_kfold import create_generators_dataframes
from run_model import compute_class_weights
import seaborn as sns
import matplotlib.pyplot as plt

# import isic dataframe
isic = import_ISIC()
class_weights = compute_class_weights(isic['class'])

# recreate 5 dataframes corresponding to dataframes used in 5-fold cv
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)

fold_no = 1

for train_index, val_index in skf.split(np.zeros(len(isic)), y=isic[['class']]):
    train_data = isic.iloc[train_index]  # create training dataframe with indices from fold split
    valid_data = isic.iloc[val_index]  # create validation dataframe with indices from fold split

    # load predictions per fold in file
    pred = pd.read_csv(
        '/Users/IrmavandenBrandt/Downloads/resnet_target=isic_source=slt10/predictions_resnet_target=isic_source'
        f'=slt10_fold{fold_no}.csv',
        header=None,
        index_col=None)

    train_datagen, valid_datagen = create_generators_dataframes(target_data='isic', augment=True)

    train_generator = train_datagen.flow_from_dataframe(dataframe=train_data,
                                                        x_col="path",
                                                        y_col="class",
                                                        target_size=(112, 112),
                                                        batch_size=128,
                                                        class_mode="categorical",
                                                        validate_filenames=False,
                                                        shuffle=True)

    valid_generator = valid_datagen.flow_from_dataframe(dataframe=valid_data,
                                                        x_col="path",
                                                        y_col="class",
                                                        target_size=(112, 112),
                                                        batch_size=128,
                                                        class_mode="categorical",
                                                        validate_filenames=False,
                                                        shuffle=False)

    # revert columns from prediction matrix to labels (labels are encoded in alphabetic order
    labels = ["AKIEC", "BBC", "BKL", "DF", "MEL", "NV", "VASC"]

    # take argmax of every row in predictions dataframe
    pred = list(pred.idxmax(axis=1))

    # create confusion matrix for all 5 folds
    cm = confusion_matrix(valid_generator.labels, pred, normalize='true')

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    plt.figure(figsize=(5.5, 4))
    sns.heatmap(cm_df, annot=True)
    plt.title(f'Confusion matrix ISIC 2018 - fold {fold_no}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    fold_no += 1


# def test_evaluation(model, generator):
#     """
#     :return: multi-class averaged AUC on test set and OneVsRest AUC scores for all individual classes
#     """
#     # compute OneVsRest multi-class macro AUC on the test set
#     OneVsRest_auc = roc_auc_score(generator.classes, model.predict(generator), multi_class='ovr',
#                                   average='macro')
#
#     # # compute individual auc scores for every class (OneVsRest)
#     clf = OneVsRestClassifier(SVC(kernel='linear', probability=True)).fit(X, y)
#     clf.fit(X_train, y_train)
#     pred = clf.predict(X_test)
#     pred_prob = clf.predict_proba(X_test)
#
#     # roc curve for classes
#     fpr = {}
#     tpr = {}
#     thresh = {}
#
#     n_class = 3
#     for i in range(n_class):
#         fpr[i], tpr[i], thresh[i] = roc_curve(y_test, pred_prob[:, i], pos_label=i)
#
#     return OneVsRest_auc

# todo: one-vs-rest: every time make one label 0 and all others 1 (or vice versa) and then compute the fpr and tpr
# use these in the roc_curve from sklearn
