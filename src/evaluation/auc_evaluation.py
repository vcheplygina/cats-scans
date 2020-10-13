# # %%
# from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, auc
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.svm import SVC
# from sklearn.preprocessing import label_binarize
# import pandas as pd
# from sklearn.model_selection import StratifiedKFold
# from src.data_import import import_ISIC
# import numpy as np
# from src.tf_generators_models_kfold import create_generators_dataframes
# from src.run_model import compute_class_weights
# import seaborn as sns
# import matplotlib.pyplot as plt
# from keras.models_base import load_model
#
#
# # import isic dataframe
# isic = import_ISIC()
# class_weights = compute_class_weights(isic['class'])
# # create list with labels in alphabetic order (they are given to model via generator in alphabetic order)
# labels = ["AKIEC", "BBC", "BKL", "DF", "MEL", "NV", "VASC"]
#
#
# def create_cm(predictions, valid_generator, labels, fold_no):
#     """
#     :param predictions:
#     :param valid_generator:
#     :param labels:
#     :param fold_no:
#     :return:
#     """
#     # take argmax of every row in predictions dataframe
#     # pred_maxed = list(predictions.idxmax(axis=1))
#     pred_maxed = np.argmax(predictions, axis=1)
#
#     # create confusion matrix for all 5 folds
#     cm = confusion_matrix(valid_generator.labels, pred_maxed, normalize='true')
#
#     cm_df = pd.DataFrame(cm, index=labels, columns=labels)
#
#     plt.figure(figsize=(5.5, 4))
#     sns.heatmap(cm_df, annot=True)
#     plt.title(f'Confusion matrix ISIC 2018 - fold {fold_no}')
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.show()
#
#
# def compute_onevsrest_auc(valid_generator, predictions):
#     # Binarize the output
#     y_true_bin = label_binarize(valid_generator.labels, classes=np.unique(valid_generator.labels))
#     pred_maxed = np.argmax(predictions, axis=1)
#     y_pred_bin = label_binarize(pred_maxed, classes=np.unique(pred_maxed))
#     n_classes = y_true_bin.shape[1]
#
#     classifier = OneVsRestClassifier(SVC(kernel='linear', probability=True, random_state=2))
#     # Floris: maak predictions op train en test set --> fit(predictions_train, y_train).decision_function(predictions_test)
#     # en dan de y_test in roc_curve
#     # todo: maak predictions met trained model op train set --> vergelijk die met predicites op validatie set
#     y_score = classifier.fit(valid_generator.labels, y_true_bin).decision_function(pred_maxed)
#
#     fpr = dict()
#     tpr = dict()
#     roc_auc = dict()
#     for i in range(n_classes):
#         fpr[i], tpr[i], _ = roc_curve(y_pred_bin[:, i], y_score[:, i])
#         roc_auc[i] = auc(fpr[i], tpr[i])
#
#     return roc_auc
#
#
# # recreate 5 dataframes corresponding to dataframes used in 5-fold cv
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
#
# fold_no = 1
#
# for train_index, val_index in skf.split(np.zeros(len(isic)), y=isic[['class']]):
#     train_data = isic.iloc[train_index]  # create training dataframe with indices from fold split
#     valid_data = isic.iloc[val_index]  # create validation dataframe with indices from fold split
#
#     # load predictions per fold in file
#     pred = pd.read_csv(
#         '/Users/IrmavandenBrandt/Downloads/resnet_target=isic_source=textures/predictions_resnet_target=isic_source='
#         f'textures_fold{fold_no}.csv',
#         header=None,
#         index_col=None)
#
#     pred_array = np.array(pred)
#     # NOTE: since we rounded the predictions to 3 digits after comma the probabilities do not always sum up to one
#     # any more (are below or above one)....
#
#     model = load_model(
#         '/Users/IrmavandenBrandt/Downloads/resnet_target=isic_source=textures/model_weights_resnet_target=isic_source'
#         f'=textures_fold{fold_no}.h5')
#
#     train_datagen, valid_datagen = create_generators_dataframes(target_data='isic', augment=True)
#
#     train_generator = train_datagen.flow_from_dataframe(dataframe=train_data,
#                                                         x_col="path",
#                                                         y_col="class",
#                                                         target_size=(112, 112),
#                                                         batch_size=128,
#                                                         class_mode="categorical",
#                                                         validate_filenames=False,
#                                                         shuffle=True)
#
#     valid_generator = valid_datagen.flow_from_dataframe(dataframe=valid_data,
#                                                         x_col="path",
#                                                         y_col="class",
#                                                         target_size=(112, 112),
#                                                         batch_size=128,
#                                                         class_mode="categorical",
#                                                         validate_filenames=False,
#                                                         shuffle=False)
#
#     valid_loss, valid_acc = model.evaluate(valid_generator, verbose=1)
#     print(valid_loss, valid_acc)
#
#     predictions = model.predict(valid_generator)
#     OneVsRest_auc = roc_auc_score(valid_generator.labels, predictions, multi_class='ovr', average='macro')
#     print(OneVsRest_auc)
#
#     create_cm(predictions, valid_generator, labels, fold_no)
#
#     # compute_onevsrest_auc(valid_generator, predictions)
#
#
#     fold_no += 1
#
#
# # def test_evaluation(model, generator):
# #     """
# #     :return: multi-class averaged AUC on test set and OneVsRest AUC scores for all individual classes
# #     """
# #     # compute OneVsRest multi-class macro AUC on the test set
# #     OneVsRest_auc = roc_auc_score(generator.classes, model.predict(generator), multi_class='ovr',
# #                                   average='macro')
# #
# #     # # compute individual auc scores for every class (OneVsRest)
# #     clf = OneVsRestClassifier(SVC(kernel='linear', probability=True)).fit(X, y)
# #     clf.fit(X_train, y_train)
# #     pred = clf.predict(X_test)
# #     pred_prob = clf.predict_proba(X_test)
# #
# #     # roc curve for classes
# #     fpr = {}
# #     tpr = {}
# #     thresh = {}
# #
# #     n_class = 3
# #     for i in range(n_class):
# #         fpr[i], tpr[i], thresh[i] = roc_curve(y_test, pred_prob[:, i], pos_label=i)
# #
# #     return OneVsRest_auc
#
#
