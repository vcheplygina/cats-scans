from sklearn.metrics import roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from src.models.model_preparation_saving import prepare_model_target


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
#                 f'{home}/output/resnet_target={target_data}_source={source_data}/model_weights_resnet_target={target_data}'
#                 f'_source={source_data}_fold{fold_no}.h5')
#             print('found model')
#         except OSError:
#             print('OSError')
#             continue
#
#         predictions = trained_model.predict(valid_generator)  # get predictions
#         print(predictions)
#         OnevsRestAUC = calculate_AUC(target_data, valid_generator, predictions)
#         auc_per_fold.append(OnevsRestAUC)
#
#         fold_no += 1
#
#     mean_auc = np.mean(auc_per_fold)
#     std_auc = np.std(auc_per_fold)
#
#     return mean_auc, std_auc
#
# #%%
# mean_auc, std_auc = collect_AUC_scores(home='/Users/IrmavandenBrandt/Downloads/Internship',
#                                        source_data='sti10', target_data='pcam-middle',
#                                        x_col='path', y_col='class',
#                                         augment=True, n_folds=5,
#                                        img_length=96, img_width=96,
#                                        batch_size=112)
# #%%
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
#     mean_auc_dict = {}
#     std_auc_dict = {}
#
#     source_datasets = ['imagenet', 'stl10', 'sti10', 'textures', 'isic', 'chest', 'pcam-middle', 'pcam-small']
#     target_datasets = ['isic', 'chest', 'pcam-middle']
#
#     for source in source_datasets:
#         print(source)
#         mean_aucs_per_source = []
#         std_aucs_per_source = []
#         for target in target_datasets:
#             print(target)
#             if target == 'pcam-middle':
#                 img_length = 96
#                 img_width = 96
#             else:
#                 img_length = 112
#                 img_width = 112
#             mean_auc, std_auc = collect_AUC_scores(home, source, target, x_col, y_col, augment, n_folds, img_length,
#                                                    img_width, batch_size)
#             mean_aucs_per_source.append(mean_auc)
#             std_aucs_per_source.append(std_auc)
#         mean_auc_dict[source] = mean_aucs_per_source
#         std_auc_dict[source] = std_aucs_per_source
#
#     return mean_auc_dict, std_auc_dict
#
#
# # %%
# mean_auc_dict, std_auc_dict = create_AUC_matrix(home='/Users/IrmavandenBrandt/Downloads/Internship', x_col='path',
#                                                 y_col='class',
#                                                 augment=True, n_folds=5, batch_size=128)
#
#
# # %%
# def create_heatmap(mean_auc_dict):
#     """
#     """
#     auc_matrix = pd.DataFrame.from_dict(auc_dict, orient='index', columns=['ISIC2018', 'Chest X-rays', 'PCam-middle'])
#     auc_matrix_nonan = auc_matrix.fillna(0)  # fill the nan-values with 0 to avoid weird ranking
#     auc_matrix_sorted = auc_matrix_nonan.apply(np.argsort, axis=0)  # sort column wise
#     auc_matrix_sorted2 = auc_matrix_sorted.apply(np.argsort, axis=0)  # sort column wise again to get correct ranking
#     # add a column that sums up all the rankings of one row to get overall ranking of source dataset
#     auc_matrix_sorted2['Overall rank'] = auc_matrix_sorted2.sum(axis=1)
#     # sort column wise to get ranking of overall rank column
#     auc_matrix_sorted3 = auc_matrix_sorted2.apply(np.argsort, axis=0)
#     auc_matrix_sorted4 = auc_matrix_sorted3.apply(np.argsort, axis=0)  # sort column wise again to get correct ranking
#     # replace the 0 values by nan so that they are not included in the heatmap
#     auc_matrix_sorted4[auc_matrix_nonan == 0] = np.nan
#
#     # create heatmap by passing ranking matrix to matplotlib
#     fig, ax = plt.subplots()
#     # add vmin so that nan values are discarded, add red-yelllow-green colormapping
#     im = ax.imshow(auc_matrix_sorted4, interpolation=None, vmin=0, aspect="auto", cmap="RdYlGn")
#     cbar = ax.figure.colorbar(im)  # add colorbar to the right of the figure, with ticks corresponding to the ranking
#     cbar.ax.set_ylabel('Rank', rotation=-90, va="bottom")
#     cbar.set_ticks(np.arange(auc_matrix_sorted2.shape[0]))
#     # ranking is from smallest integer to largest integer (i.e. 0 - ... ) so add 1 to every integer of the ranking
#     tick_labels = [i + 1 for i in range(auc_matrix_sorted2.shape[0])]
#     # reverse the tick labels to get tick labels that makes sense (ranking 1 is best, 2 is second best ...)
#     cbar.set_ticklabels(tick_labels[::-1])
#     # set ticks with labels to the heatmap corresponding to the number of rows and columns in the ranking dataframe
#     ax.set_xticks(np.arange(auc_matrix_sorted2.shape[1]))
#     ax.set_yticks(np.arange(auc_matrix_sorted2.shape[0]))
#     ax.set_xticklabels(auc_matrix_sorted2.axes[1], rotation=45)
#     ax.set_yticklabels(auc_matrix_sorted2.axes[0])
#     # add white lines in between rectangles to make figure look nicer
#     for edge, spine in ax.spines.items():
#         spine.set_visible(False)
#     ax.set_xticks(np.arange(auc_matrix.shape[1] + 1) - .5, minor=True)
#     ax.set_yticks(np.arange(auc_matrix.shape[0] + 1) - .5, minor=True)
#     ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
#     ax.tick_params(which="minor", bottom=False, left=False)
#
#     # add AUC-scores in heatmap by going over rows and column of auc_matrix_nonan
#     j = 0
#     for index, rows in auc_matrix_nonan.iterrows():
#         i = 0
#         for i in range(auc_matrix_nonan.shape[1]):
#             if rows[i] == 0.0:
#                 continue
#             text = ax.text(i, j, rows[i], ha="center", va="center")
#             i += 1
#         j += 1
#
#     # add labels to axes, and create a tight layout so that png file is not cut on sides
#     plt.xlabel('Target')
#     plt.ylabel('Source')
#     plt.rcParams["axes.labelsize"] = 12
#     plt.tight_layout()
#     plt.savefig('outputs/heatmap_auc_scores', dpi=1000)  # save figure with extra resolution
#     plt.show()
#
#
# def create_barplot(mean_auc_dict, std_auc_dict):
#
#     labels = ['ISIC2018', 'Chest X-rays', 'PCam-middle']
#     imagenet_scores = [0.947, 0.985, 0.961]
#     imagenet_error = [0.003, 0.003, 0.002]
#     stl10_scores = [0.905, 0.965, 0.958]
#     stl10_error = [0.004, 0.001, 0.001]
#     sti10_scores = [0.895, 0.954, 0.879]
#     sti10_error = [0.006, 0.002, 0.003]
#     dtd_scores = [0.910, 0.981, 0.925]
#     dtd_error = [0.004, 0.002, 0.001]
#     isic_scores = [np.nan, 0.847, 0.921]
#     isic_error = [np.nan, 0.002, 0.016]
#     chest_scores = [0.900, np.nan, 0.811]
#     chest_error = [0.008, np.nan, 0.004]
#     pcam_mid_scores = [0.915, 0.958, np.nan]
#     pcam_mid_error = [0.006, 0.005, np.nan]
#     pcam_small_scores = [0.901, 0.949, np.nan]
#     pcam_small_error = [0.007, 0.012, np.nan]
#
#     x = np.arange(len(labels))  # the label locations
#     width = 0.08  # the width of the bars
#
#     fig, ax = plt.subplots()
#     ax.grid(zorder=0)
#     rects1 = ax.bar(x - 4*width, imagenet_scores, width, yerr=imagenet_error, ecolor='black',
#                     capsize=2, label='ImageNet', zorder=5, color=(0.4, 0.7607843137254902, 0.6470588235294118, 1.0))
#     rects2 = ax.bar(x - 3*width, stl10_scores, width, yerr=stl10_error, ecolor='black',
#                     capsize=2, label='STL-10', zorder=3, color=(0.9882352941176471, 0.5529411764705883, 0.3843137254901961, 1.0))
#     rects3 = ax.bar(x - 2*width, sti10_scores, width, yerr=sti10_error, ecolor='black',
#                     capsize=2, label='STI-10', zorder=3, color=(0.5529411764705883, 0.6274509803921569, 0.796078431372549, 1.0))
#     rects4 = ax.bar(x - width, dtd_scores, width, yerr=dtd_error, ecolor='black',
#                     capsize=2, label='DTD', zorder=3, color=(0.9058823529411765, 0.5411764705882353, 0.7647058823529411, 1.0))
#     rects5 = ax.bar(x, isic_scores, width, yerr=isic_error, ecolor='black',
#                     capsize=2, label='ISIC2018', zorder=3, color=(0.6509803921568628, 0.8470588235294118, 0.32941176470588235, 1.0))
#     rects6 = ax.bar(x + width, chest_scores, width, yerr=chest_error, ecolor='black',
#                     capsize=2, label='Chest X-rays', zorder=3, color=(1.0, 0.8509803921568627, 0.1843137254901961, 1.0))
#     rects7 = ax.bar(x + 2*width, pcam_mid_scores, width, yerr=pcam_mid_error, ecolor='black',
#                     capsize=2, label='PCam-middle', zorder=3, color=(0.8980392156862745, 0.7686274509803922, 0.5803921568627451, 1.0))
#     rects8 = ax.bar(x + 3*width, pcam_small_scores, width, yerr=pcam_small_error, ecolor='black',
#                     capsize=2, label='PCam-small', zorder=3, color=(0.7019607843137254, 0.7019607843137254, 0.7019607843137254, 1.0))
#
#     # Add some text for labels, title and custom x-axis tick labels, etc.
#     ax.set_ylabel('AUC-score')
#     ax.set_xlabel('Target Dataset')
#     ax.set_title('AUC-score by Source and Target Dataset')
#     ax.set_xticks(x)
#     ax.set_xticklabels(labels)
#     plt.ylim(0.8)
#     plt.legend(loc="upper right", bbox_to_anchor=(1, 1), ncol=3)
#     plt.rcParams['font.family'] = 'sans-serif'
#     plt.rcParams['font.sans-serif'] = 'Helvetica'
#     plt.savefig('outputs/barplot_auc_scores', dpi=1000)
#
#     plt.show()

# %%


# #%%
# auc_scores = np.array([[0.947, 0.985, 0.961], [0.905, 0.965, 0.958], [0.895, 0.954, 0.879], [0.910, 0.981, 0.925],
#                        [0.905, 0.965, 0.958], [0.895, 0.954, 0.879], [0.915, 0.958, np.nan], [np.nan, 0.921, 0.847],
#                        [0.900, np.nan, 0.900], [0.915, 0.958, np.nan], [0.901, 0.949, np.nan]])
#
# np.save('outputs/mean_auc_scores', auc_scores)
# %%


# %%
# # create dictionary with auc scores, NOTE: for argsort nan values are not allowed so fill those in with 0
# auc_scores = {'ImageNet': [0.947, 0.985, 0.961], 'STL-10': [0.905, 0.965, 0.958], 'STI-10': [0.895, 0.954, 0.879],
#               'DTD': [0.910, 0.981, 0.925], 'ISIC2018': [np.nan, 0.921, 0.847], 'Chest X-rays': [0.900, np.nan, 0.811],
#               'PCam-middle': [0.915, 0.958, np.nan], 'PCam-small': [0.901, 0.949, np.nan]}
# auc_matrix = pd.DataFrame.from_dict(auc_scores, orient='index', columns=['ISIC2018', 'Chest X-rays', 'PCam-middle'])
# auc_matrix_nonan = auc_matrix.fillna(0)
# auc_matrix_sorted = auc_matrix_nonan.apply(np.argsort, axis=0)  # sort column wise
# auc_matrix_sorted2 = auc_matrix_sorted.apply(np.argsort, axis=0)  # sort column wise again
# auc_matrix_sorted2['Overall rank'] = auc_matrix_sorted2.sum(axis=1)
# auc_matrix_sorted3 = auc_matrix_sorted2.apply(np.argsort, axis=0)  # sort column wise again
# auc_matrix_sorted4 = auc_matrix_sorted3.apply(np.argsort, axis=0)  # sort column wise again
#
# auc_matrix_sorted4[auc_matrix_nonan == 0] = np.nan  # replace the 0 values by nan
#
# fig, ax = plt.subplots()
# im = ax.imshow(auc_matrix_sorted4, interpolation=None, vmin=0, aspect="auto", cmap="RdYlGn")
# cbar = ax.figure.colorbar(im)
# cbar.ax.set_ylabel('Rank', rotation=-90, va="bottom")
# cbar.set_ticks(np.arange(auc_matrix_sorted2.shape[0]))
# tick_labels = [i + 1 for i in range(auc_matrix_sorted2.shape[0])]
# cbar.set_ticklabels(tick_labels[::-1])
# ax.set_xticks(np.arange(auc_matrix_sorted2.shape[1]))
# ax.set_yticks(np.arange(auc_matrix_sorted2.shape[0]))
# ax.set_xticklabels(auc_matrix_sorted2.axes[1], rotation=45)
# ax.set_yticklabels(auc_matrix_sorted2.axes[0])
# for edge, spine in ax.spines.items():
#     spine.set_visible(False)
# ax.set_xticks(np.arange(auc_matrix.shape[1] + 1) - .5, minor=True)
# ax.set_yticks(np.arange(auc_matrix.shape[0] + 1) - .5, minor=True)
# ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
# ax.tick_params(which="minor", bottom=False, left=False)
#
# j = 0
# for index, rows in auc_matrix_nonan.iterrows():
#     i = 0
#     for i in range(auc_matrix_nonan.shape[1]):
#         if rows[i] == 0.0:
#             continue
#         text = ax.text(i, j, rows[i], ha="center", va="center")
#         i += 1
#     j += 1
#
# plt.xlabel('Target')
# plt.ylabel('Source')
# plt.rcParams["axes.labelsize"] = 12
# plt.tight_layout()
# # plt.savefig('outputs/heatmap_auc_scores', dpi=1000)
# plt.show()

#
#
# #%%
# import matplotlib.pyplot as plt
# import numpy as np
#
# #
# # labels = ['ImageNet', 'STL-10', 'STI-10', 'DTD', 'ISIC2018', 'Chest X-rays', 'PCam-middle', 'PCam-small']
# # isic_scores = [0.947, 0.905, 0.895, 0.910, np.nan, 0.900, 0.915, 0.901]
# # isic_error = [0.003, 0.004, 0.006, 0.004, np.nan, 0.008, 0.006, 0.007]
# # chest_scores = [0.985, 0.965, 0.954, 0.981, 0.921, np.nan, 0.958, 0.949]
# # chest_error = [0.003, 0.001, 0.002, 0.002, 0.016, np.nan, 0.005, 0.012]
# # pcam_scores = [0.961, 0.958, 0.879, 0.925, 0.847, 0.811, np.nan, np.nan]
# # pcam_error = [0.002, 0.001, 0.003, 0.001, 0.002, 0.004, np.nan, np.nan]
# #
# # x = np.arange(len(labels))  # the label locations
# # width = 0.2  # the width of the bars
# #
# # fig, ax = plt.subplots()
# # rects1 = ax.bar(x - width, isic_scores, width, yerr=isic_error, alpha=0.8, ecolor='black', capsize=2, label='ISIC2018')
# # rects2 = ax.bar(x, chest_scores, width, yerr=chest_error, alpha=0.8, ecolor='black', capsize=2, label='Chest X-rays')
# # rects3 = ax.bar(x + width, pcam_scores, width, yerr=pcam_error, alpha=0.8, ecolor='black', capsize=2, label='PCam-middle')
#
#
# labels = ['ISIC2018', 'Chest X-rays', 'PCam-middle']
# imagenet_scores = [0.947, 0.985, 0.961]
# imagenet_error = [0.003, 0.003, 0.002]
# stl10_scores = [0.905, 0.965, 0.958]
# stl10_error = [0.004, 0.001, 0.001]
# sti10_scores = [0.895, 0.954, 0.879]
# sti10_error = [0.006, 0.002, 0.003]
# dtd_scores = [0.910, 0.981, 0.925]
# dtd_error = [0.004, 0.002, 0.001]
# isic_scores = [np.nan, 0.847, 0.921]
# isic_error = [np.nan, 0.002, 0.016]
# chest_scores = [0.900, np.nan, 0.811]
# chest_error = [0.008, np.nan, 0.004]
# pcam_mid_scores = [0.915, 0.958, np.nan]
# pcam_mid_error = [0.006, 0.005, np.nan]
# pcam_small_scores = [0.901, 0.949, np.nan]
# pcam_small_error = [0.007, 0.012, np.nan]
#
# x = np.arange(len(labels))  # the label locations
# width = 0.08  # the width of the bars
#
# fig, ax = plt.subplots()
# ax.grid(zorder=0)
# rects1 = ax.bar(x - 4*width, imagenet_scores, width, yerr=imagenet_error, ecolor='black',
#                 capsize=2, label='ImageNet', zorder=5, color=(0.4, 0.7607843137254902, 0.6470588235294118, 1.0))
# rects2 = ax.bar(x - 3*width, stl10_scores, width, yerr=stl10_error, ecolor='black',
#                 capsize=2, label='STL-10', zorder=3, color=(0.9882352941176471, 0.5529411764705883, 0.3843137254901961, 1.0))
# rects3 = ax.bar(x - 2*width, sti10_scores, width, yerr=sti10_error, ecolor='black',
#                 capsize=2, label='STI-10', zorder=3, color=(0.5529411764705883, 0.6274509803921569, 0.796078431372549, 1.0))
# rects4 = ax.bar(x - width, dtd_scores, width, yerr=dtd_error, ecolor='black',
#                 capsize=2, label='DTD', zorder=3, color=(0.9058823529411765, 0.5411764705882353, 0.7647058823529411, 1.0))
# rects5 = ax.bar(x, isic_scores, width, yerr=isic_error, ecolor='black',
#                 capsize=2, label='ISIC2018', zorder=3, color=(0.6509803921568628, 0.8470588235294118, 0.32941176470588235, 1.0))
# rects6 = ax.bar(x + width, chest_scores, width, yerr=chest_error, ecolor='black',
#                 capsize=2, label='Chest X-rays', zorder=3, color=(1.0, 0.8509803921568627, 0.1843137254901961, 1.0))
# rects7 = ax.bar(x + 2*width, pcam_mid_scores, width, yerr=pcam_mid_error, ecolor='black',
#                 capsize=2, label='PCam-middle', zorder=3, color=(0.8980392156862745, 0.7686274509803922, 0.5803921568627451, 1.0))
# rects8 = ax.bar(x + 3*width, pcam_small_scores, width, yerr=pcam_small_error, ecolor='black',
#                 capsize=2, label='PCam-small', zorder=3, color=(0.7019607843137254, 0.7019607843137254, 0.7019607843137254, 1.0))
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('AUC-score')
# ax.set_xlabel('Target Dataset')
# ax.set_title('AUC-score by Source and Target Dataset')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# plt.ylim(0.8)
# plt.legend(loc="upper right", bbox_to_anchor=(1, 1), ncol=3)
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = 'Helvetica'
# plt.savefig('outputs/barplot_auc_scores', dpi=1000)
#
# plt.show()
