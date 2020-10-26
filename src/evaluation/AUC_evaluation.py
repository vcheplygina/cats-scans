# from sklearn.metrics import roc_auc_score
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# from keras.models import load_model
# from ..io.data_import import collect_data
# from ..models.model_preparation_saving import prepare_model_target
#
#
# def calculate_AUC(target_data, valid_generator, predictions):
#     """
#     :param target_data: dataset used as target dataset
#     :param valid_generator: generator feeding validation images and labels to model
#     :param predictions: predictions made on the validation set using the trained model
#     :return: One-Vs-Rest AUC for multiclass case, 'normal' AUC for binary case
#     """
#     # compute OneVsRest multi-class macro AUC on the test set
#     if target_data == "isic":
#         OneVsRest_auc = roc_auc_score(valid_generator.classes, predictions, multi_class='ovr', average='macro')
#     else:
#         OneVsRest_auc = roc_auc_score(valid_generator.classes, predictions, average='macro')
#     print(f'Validation auc: {OneVsRest_auc}')
#
#     return OneVsRest_auc
#
#
# def collect_AUC_scores(home, source_data, target_data, x_col, y_col, augment, n_folds, img_length, img_width,
#                        batch_size):
#     """
#     :param home:
#     :param source_data:
#     :param target_data:
#     :param x_col:
#     :param y_col:
#     :param augment:
#     :param n_folds:
#     :param img_length:
#     :param img_width:
#     :param batch_size:
#     :return:
#     """
#     num_classes, dataframe, skf, train_datagen, valid_datagen, x_col, y_col, class_mode = prepare_model_target(home,
#                                                                                                                target_data,
#                                                                                                                source_data,
#                                                                                                                x_col,
#                                                                                                                y_col,
#                                                                                                                augment,
#                                                                                                                n_folds)
#     auc_per_fold = []
#     fold_no = 1
#
#     for train_index, val_index in skf.split(np.zeros(len(dataframe)), y=dataframe[['class']]):
#         print(f'Starting fold {fold_no}')
#
#         valid_data = dataframe.iloc[val_index]  # create validation dataframe with indices from fold split
#
#         valid_generator = valid_datagen.flow_from_dataframe(dataframe=valid_data,
#                                                             x_col=x_col,
#                                                             y_col=y_col,
#                                                             target_size=(img_length, img_width),
#                                                             batch_size=batch_size,
#                                                             class_mode=class_mode,
#                                                             validate_filenames=False,
#                                                             shuffle=False)
#
#         try:
#             trained_model = load_model(
#                 f'/output/model_weights_resnet_target={target_data}_source={source_data}_fold{i}.h5')
#         except:
#             continue
#
#         predictions = trained_model.predict(valid_generator)  # get predictions
#         OnevsRestAUC = calculate_AUC(target_data, valid_generator, predictions)
#         auc_per_fold.append(OnevsRestAUC)
#
#     mean_auc = np.mean(auc_per_fold)
#
#     return mean_auc
#
#
# def create_AUC_matrix(home, x_col, y_col, augment, n_folds, batch_size):
#     """
#     :param home:
#     :param x_col:
#     :param y_col:
#     :param augment:
#     :param n_folds:
#     :param batch_size:
#     :return:
#     """
#     auc_dict = {}
#
#     source_datasets = ['imagenet', 'stl10', 'sti10', 'textures', 'isic', 'chest', 'pcam-middle', 'pcam-small']
#     target_datasets = ['isic', 'chest', 'pcam-middle']
#
#     for source in source_datasets:
#         aucs_per_source = []
#         for target in target_datasets:
#             if target == 'pcam-middle':
#                 img_length = 96
#                 img_width = 96
#             else:
#                 img_length = 112
#                 img_width = 112
#             mean_auc = collect_AUC_scores(home, source, target, x_col, y_col, augment, n_folds, img_length,
#                                           img_width, batch_size)
#             aucs_per_source.append(mean_auc)
#         auc_dict[source] = aucs_per_source
#
#     return auc_dict
#
#
# auc_dictionary = create_AUC_matrix(home='/Users/IrmavandenBrandt/Downloads/Internship', x_col='path', y_col='class',
#                                    augment=True, n_folds=5, batch_size=128)
#
# #
# #
# # # def viz_AUC_heatmap(source_data, target_data, ):
# #
# #
# #
# # # %%
# # # create dictionary with auc scores, NOTE: for argsort nan values are not allowed so fill those in with 0
# # auc_scores = {'ImageNet': [0.947, 0.985, 0.961], 'STL-10': [0.905, 0.965, 0.958], 'STI-10': [0.895, 0.954, 0.879],
# #               'DTD': [0.910, 0.981, 0.925], 'ISIC2018': [np.nan, 0.921, 0.847], 'Chest X-rays': [0.900, np.nan, 0.900],
# #               'PCam-middle': [0.915, 0.958, np.nan], 'PCam-small': [0.901, 0.949, np.nan]}
# # auc_matrix = pd.DataFrame.from_dict(auc_scores, orient='index', columns=['ISIC2018', 'Chest X-rays', 'PCam-middle'])
# # auc_matrix_nonan = auc_matrix.fillna(0)
# # auc_matrix_sorted = auc_matrix_nonan.apply(np.argsort, axis=0)  # sort column wise
# # auc_matrix_sorted2 = auc_matrix_sorted.apply(np.argsort, axis=0)  # sort column wise again
# # auc_matrix_sorted2[auc_matrix_nonan == 0] = np.nan  # replace the 0 values by nan
# #
# # fig, ax = plt.subplots()
# # im = ax.imshow(auc_matrix_sorted2, interpolation=None, vmin=0, aspect="auto", cmap="RdYlGn")
# # cbar = ax.figure.colorbar(im, )
# # # cbar.ax.set_ylabel(cbarlabel='AUC-score', rotation=-90, va="bottom")
# # cbar.set_ticks([0, 3.5, 7])
# # cbar.set_ticklabels([np.min(np.ma.masked_array(auc_matrix.to_numpy(), np.isnan(auc_matrix.to_numpy()), axis=1)),
# #                      np.nanquantile(auc_matrix.to_numpy(), 0.5),
# #                      np.max(np.ma.masked_array(auc_matrix.to_numpy(), np.isnan(auc_matrix.to_numpy()), axis=1))])
# #
# # ax.set_xticks(np.arange(auc_matrix.shape[1]))
# # ax.set_yticks(np.arange(auc_matrix.shape[0]))
# # ax.set_xticklabels(auc_matrix.axes[1])
# # ax.set_yticklabels(auc_matrix.axes[0])
# # for edge, spine in ax.spines.items():
# #     spine.set_visible(False)
# # ax.set_xticks(np.arange(auc_matrix.shape[1] + 1) - .5, minor=True)
# # ax.set_yticks(np.arange(auc_matrix.shape[0] + 1) - .5, minor=True)
# # ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
# # ax.tick_params(which="minor", bottom=False, left=False)
# #
# # j = 0
# # for index, rows in auc_matrix_nonan.iterrows():
# #     i = 0
# #     for i in range(auc_matrix_nonan.shape[1]):
# #         if rows[i] == 0.0:
# #             continue
# #         text = ax.text(i, j, rows[i], ha="center", va="center")
# #         i += 1
# #     j += 1
# #
# # plt.xlabel('Target')
# # plt.ylabel('Source')
# # plt.rcParams["axes.labelsize"] = 12
# #
# # plt.show()
