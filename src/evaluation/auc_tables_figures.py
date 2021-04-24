from src.evaluation.auc_evaluation import calculate_AUC
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from src.models.model_preparation_saving import prepare_model_target, prepare_model_source
from numpy.random import seed
import tensorflow as tf
import csv

seed(1)
tf.random.set_seed(2)


def calculate_pretrain_acc_AUC(home, source_data, target_data, augment, batch_size, img_length, img_width):
    """
    :param home: part of path that is specific to user, e.g. /Users/..../
    :param source_data: dataset used as source dataset
    :param target_data: dataset used as target dataset
    :param augment: boolean specifying whether to use data augmentation or not
    :param batch_size: number of images processed in one batch
    :param img_length: target length of image in pixels
    :param img_width: target width of image in pixels
    :return : pretraining accuracy scores on training, validation and test set, AUC score on test set
    """
    num_classes, train_generator, valid_generator, test_generator = prepare_model_source(home,
                                                                                         source_data,
                                                                                         target_data,
                                                                                         augment,
                                                                                         batch_size,
                                                                                         img_length,
                                                                                         img_width)

    trained_model = load_model(f'{home}/pretrain_models/model_weights_resnet_pretrained={source_data}.h5')
    print('found model')

    # compute loss and accuracy on training, validation and test set
    train_loss, train_acc = trained_model.evaluate(train_generator, verbose=1)
    print(f'Train loss:', train_loss, f' and Train accuracy:', train_acc)
    val_loss, val_acc = trained_model.evaluate(valid_generator, verbose=1)
    print(f'Validation loss:', val_loss, f' and Validation accuracy:', val_acc)
    test_loss, test_acc = trained_model.evaluate(test_generator, verbose=1)
    print(f'Test loss:', test_loss, f' and Test accuracy:', test_acc)

    predictions = trained_model.predict(test_generator)  # get predictions
    auc = calculate_AUC(source_data, test_generator, predictions)  # calculate AUC for test set

    return train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, auc


def collect_TF_AUC(home, source_data, target_data, x_col, y_col, augment, k, batch_size, img_length, img_width):
    """
    :param home: part of path that is specific to user, e.g. /Users/..../
    :param source_data: dataset used as source dataset
    :param target_data: dataset used as target dataset
    :param x_col: column in dataframe containing the image paths
    :param y_col: column in dataframe containing the target labels
    :param augment: boolean specifying whether to use data augmentation or not
    :param k: amount of folds used in the k-fold cross validation
    :param batch_size: number of images processed in one batch
    :param img_length: target length of image in pixels
    :param img_width: target width of image in pixels
    :return: mean and standard deviation of AUC score for k-fold cross validation transfer learning experiment
    """
    dataframe, num_classes, x_col, y_col, class_mode, skf, train_gen, valid_gen = prepare_model_target(home,
                                                                                                       source_data,
                                                                                                       target_data,
                                                                                                       x_col,
                                                                                                       y_col,
                                                                                                       augment,
                                                                                                       k)

    auc_per_fold = []
    auc_per_fold_dict = {}
    fold_no = 1

    for train_index, val_index in skf.split(np.zeros(len(dataframe)), y=dataframe[['class']]):
        print(f'Starting fold {fold_no}')

        valid_data = dataframe.iloc[val_index]  # create validation dataframe with indices from fold split

        valid_generator = valid_gen.flow_from_dataframe(dataframe=valid_data,
                                                        x_col=x_col,
                                                        y_col=y_col,
                                                        target_size=(img_length, img_width),
                                                        batch_size=batch_size,
                                                        class_mode=class_mode,
                                                        validate_filenames=False,
                                                        seed=2,
                                                        shuffle=False)

        try:
            trained_model = load_model(
                f'{home}/output/resnet_target={target_data}_source={source_data}/model_weights_resnet_target='
                f'{target_data}_source={source_data}_fold{fold_no}.h5')
            print('found model')
        except OSError:
            print('OSError')
            continue

        predictions = trained_model.predict(valid_generator)  # get predictions
        auc = calculate_AUC(target_data, valid_generator, predictions)  # compute AUC score
        auc_per_fold.append(auc)
        auc_per_fold_dict[fold_no] = auc

        fold_no += 1

    mean_auc = np.mean(auc_per_fold)
    std_auc = np.std(auc_per_fold)

    return mean_auc, std_auc, auc_per_fold_dict


def create_AUC_matrix(home, x_col, y_col, augment, k, batch_size):
    """
    :param home: part of path that is specific to user, e.g. /Users/..../
    :param x_col: column in dataframe containing the image paths
    :param y_col: column in dataframe containing the target labels
    :param augment: boolean specifying whether to use data augmentation or not
    :param k: amount of folds used in the k-fold cross validation
    :param batch_size: number of images processed in one batch
    :return: dictionary containing all mean AUC scores from transfer learning, dictionary containing all standard
    deviations of the AUC scores from transfer learning
    """
    mean_auc_dict = {}
    std_auc_dict = {}
    auc_fold_dict = {}

    source_datasets = ['imagenet', 'stl10', 'sti10', 'textures', 'isic', 'chest', 'pcam-middle', 'pcam-small', 'kimia']
    target_datasets = ['isic', 'chest', 'pcam-middle']

    for source in source_datasets:
        print(f'Now starting with source dataset {source}')
        mean_aucs_per_source = []
        std_aucs_per_source = []
        auc_folds_per_source = []
        for target in target_datasets:
            print(f'Now starting with target dataset {target}')
            if (target == source) and (target != 'isic'):
                mean_auc, std_auc = np.nan, np.nan
            else:
                # add if-else to set right img length and width depending on target dataset used
                if target == 'pcam-middle':
                    img_length = 96
                    img_width = 96
                else:
                    img_length = 112
                    img_width = 112
                mean_auc, std_auc, auc_per_fold_dict = collect_TF_AUC(home, source, target, x_col, y_col, augment, k,
                                                                      batch_size, img_length, img_width)
            mean_aucs_per_source.append(mean_auc)
            std_aucs_per_source.append(std_auc)
            auc_folds_per_source.append(auc_per_fold_dict)
        mean_auc_dict[source] = mean_aucs_per_source
        std_auc_dict[source] = std_aucs_per_source
        auc_fold_dict[source] = auc_folds_per_source

    return mean_auc_dict, std_auc_dict, auc_fold_dict


def create_heatmap(mean_auc_dict, overall_ranking):
    """
    :param mean_auc_dict: dictionary containing mean auc scores from k-fold CV transfer learning experiments
    :param overall_ranking: boolean specifying whether or not to compute overall ranking of source datasets (i.e. for
    target datasets together)
    :return : heatmap that shows ranking of source datasets per target dataset
    """
    print(mean_auc_dict)
    auc_matrix = pd.DataFrame.from_dict(mean_auc_dict, orient='index',
                                        columns=['ISIC2018', 'Chest X-rays', 'PCam-middle'])
    auc_matrix = auc_matrix.round(3)  # round all values to 3 numbers after decimal
    auc_matrix_nonan = auc_matrix.fillna(0)  # fill the nan-values with 0 to avoid weird ranking
    auc_matrix_sorted = auc_matrix_nonan.apply(np.argsort, axis=0)  # sort column wise
    auc_matrix_sorted = auc_matrix_sorted.apply(np.argsort, axis=0)  # sort column wise again to get correct ranking
    auc_matrix_sorted[auc_matrix_nonan == 0] = np.nan
    # subtract 2 from all values in the ranking since every column contains 2 missing values
    auc_matrix_sorted = auc_matrix_sorted - 2

    if overall_ranking:
        # add a column that sums up all the rankings of one row to get overall ranking of source dataset
        auc_matrix_sorted['Overall rank'] = auc_matrix_sorted.sum(axis=1) / auc_matrix_sorted.count(axis=1)
        auc_matrix_sorted = auc_matrix_sorted.fillna(0)  # fill the nan-values with 0 to avoid weird ranking
        # sort column wise to get ranking of overall rank column
        auc_matrix_sorted = auc_matrix_sorted.apply(np.argsort, axis=0)
        auc_matrix_sorted = auc_matrix_sorted.apply(np.argsort, axis=0)  # sort again to get correct ranking
        # replace the 0 values by nan so that they are not included in the heatmap
        auc_matrix_sorted[auc_matrix_nonan == 0] = np.nan

    # create heatmap by passing ranking matrix to matplotlib
    fig, ax = plt.subplots()
    # add vmin so that nan values are discarded, add red-yelllow-green colormapping
    im = ax.imshow(auc_matrix_sorted, interpolation=None, vmin=0, aspect="auto", cmap="RdYlGn")
    cbar = ax.figure.colorbar(im)  # add color bar to the right of the figure, with ticks corresponding to the ranking
    cbar.ax.set_ylabel('Rank', rotation=-90, va="bottom")
    # every column misses 2 values so ranking should go from 1 to len(column) - 2
    cbar.set_ticks(np.arange(auc_matrix_sorted.shape[0] - 2))
    # create tick labels depending on the number of source datasets, subtract -1 of every value to start at 0
    tick_labels = [i - 1 for i in range(auc_matrix_sorted.shape[0])]
    # reverse the tick labels to get tick labels that makes sense (ranking 1 is best, 2 is second best ...)
    cbar.set_ticklabels(tick_labels[::-1])
    # set ticks with labels to the heatmap corresponding to the number of rows and columns in the ranking dataframe
    ax.set_xticks(np.arange(auc_matrix_sorted.shape[1]))
    ax.set_yticks(np.arange(auc_matrix_nonan.shape[0]))
    ax.set_xticklabels(auc_matrix_sorted.axes[1], rotation=45)
    ax.set_yticklabels(['ImageNet', 'STL-10', 'STI-10', 'DTD', 'ISIC2018', 'Chest X-rays', 'PCam-middle', 'PCam-small',
                        'KimiaPath960'])
    # add white lines in between rectangles to make figure look nicer
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    ax.set_xticks(np.arange(auc_matrix.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(auc_matrix.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    # add AUC-scores in heatmap by going over rows and column of auc_matrix_nonan
    j = 0
    for index, rows in auc_matrix_nonan.iterrows():
        for i in range(auc_matrix_nonan.shape[1]):
            if rows[i] == 0.0:
                continue
            ax.text(i, j, rows[i], ha="center", va="center")
            i += 1
        j += 1

    # add labels to axes, and create a tight layout so that png file is not cut on sides
    plt.xlabel('Target')
    plt.ylabel('Source')
    plt.rcParams["axes.labelsize"] = 12
    plt.tight_layout()
    plt.savefig('outputs/heatmap_auc_scores', dpi=1000)  # save figure with extra resolution
    plt.show()


def create_barplot(mean_auc_dict, std_auc_dict):
    """
    :param mean_auc_dict: dictionary containing mean auc scores from k-fold CV transfer learning experiments
    :param std_auc_dict: dictionary containing std deviation of auc scores from k-fold CV transfer learning experiments
    :return : barplot with mean auc scores as bars and standard deviations as error bars
    """
    target = ['ISIC2018', 'Chest X-rays', 'PCam-middle']
    source = ['ImageNet', 'STL-10', 'STI-10', 'DTD', 'ISIC2018', 'Chest X-rays', 'PCam-middle', 'PCam-small',
              'KimiaPath960']

    x = np.arange(len(target))  # the label locations
    width = 0.08  # set width of all bars

    fig, ax = plt.subplots()
    ax.grid(zorder=0)  # activate grid and place it behind the bars

    # colors based on Set2 colormap of matplotlib
    colors = [(0.4, 0.7607843137254902, 0.6470588235294118, 1.0),
              (0.9882352941176471, 0.5529411764705883, 0.3843137254901961, 1.0),
              (0.5529411764705883, 0.6274509803921569, 0.796078431372549, 1.0),
              (0.9058823529411765, 0.5411764705882353, 0.7647058823529411, 1.0),
              (0.6509803921568628, 0.8470588235294118, 0.32941176470588235, 1.0),
              (1.0, 0.8509803921568627, 0.1843137254901961, 1.0),
              (0.8980392156862745, 0.7686274509803922, 0.5803921568627451, 1.0),
              (0.7019607843137254, 0.7019607843137254, 0.7019607843137254, 1.0),
              (0.8, 0.3, 0.3, 1.0)]
    # set widths corresponing to number of source datasets that are used to specify location of bars
    widths = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    # loop over all values in the dictionaries in parallel and add them to the bars and error bars of plot
    i = 0
    for (key1, value1), (key2, value2) in zip(mean_auc_dict.items(), std_auc_dict.items()):
        ax.bar(x + widths[i] * width, value1, width, yerr=value2, ecolor='black', capsize=2, label=source[i],
               zorder=5, color=colors[i])  # use a zorder bigger than grid to make sure bars are placed in front of grid
        i += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('AUC-score')
    ax.set_xlabel('Target Dataset')
    ax.set_title('AUC-score by Source and Target Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(target)
    plt.ylim(0.8)
    plt.legend(loc="upper right", bbox_to_anchor=(1, 1), ncol=3)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Helvetica'
    plt.savefig('outputs/barplot_auc_scores', dpi=1000)

    plt.show()


# collect the mean and standard deviation AUC dictionaries
mean_auc_dictionary, std_auc_dictionary, auc_fold_dictionary = create_AUC_matrix(
    home='/Users/IrmavandenBrandt/Downloads/Internship',
    x_col='path', y_col='class', augment=True, k=5,
    batch_size=128)

# create a csv file containing the auc scores per folds, the mean and std for all combinations of target and source daya
with open('results/auc_folds_means_std.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    header = ['source', 'target', 'fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5', 'mean', 'std']
    writer.writerow(header)
    rows = []
    for key, value in auc_fold_dictionary.items():
        targets = ['isic', 'chest', 'pcam-middle']
        for dicts, target in zip(value, targets):
            row = [key, target]
            values = []
            for auc_value in dicts.values():
                row.append(auc_value)
            rows.append(row)
    # next step: add mean aucs to csv
    mean_auc_values = []
    for values in mean_auc_dictionary.values():
        for mean_auc in values:
            mean_auc_values.append(mean_auc)
    for auc_value, row in zip(mean_auc_values, rows):
        row.append(auc_value)
    # final step: add std aucs to csv
    std_auc_values = []
    for values in std_auc_dictionary.values():
        for std_auc in values:
            std_auc_values.append(std_auc)
    for std_auc, row in zip(std_auc_values, rows):
        row.append(std_auc)
        writer.writerow(row)

# write the mean auc scores to a numpy array and save in outputs for part Bas
auc_scores = []
for key, value in mean_auc_dictionary.items():
    scores = np.array(value)
    auc_scores.append(scores)
np.save('outputs/mean_auc_scores', auc_scores)

# create heatmap and barplot which are saved in outputs folder
create_heatmap(mean_auc_dict=mean_auc_dictionary, overall_ranking=False)
create_barplot(mean_auc_dict=mean_auc_dictionary, std_auc_dict=std_auc_dictionary)
