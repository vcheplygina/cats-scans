# %%
from sklearn.metrics import roc_auc_score
from TF_EfficientNet import run_efficientnet


def test_evaluation(isic, blood, img_dir, label_dir, test_size, x_col, y_col, augment, validation_split, learning_rate,
                    img_length, img_width, batch_size, epochs, color, dropout, imagenet):
    """
    :param isic: boolean specifying whether data needed is ISIC data or not
    :param blood: boolean specifying whether data needed is blood data or not
    :param img_dir: directory where images are found
    :param label_dir: directory where labels are found
    :param test_size: split value used to split part of dataframe into test set
    :param x_col: column in dataframe containing the image paths
    :param y_col: column in dataframe containing the target labels
    :param augment: boolean specifying whether to use data augmentation or not
    :param validation_split: fraction of images from training set used as validation set
    :param img_length: target length of image in pixels
    :param img_width: target width of image in pixels
    :param batch_size: amount of images processed per batch
    :param learning_rate: learning rate used by optimizer
    :param epochs: number of epochs the model needs to run
    :param color: boolean specifying whether the images are in color or not
    :param dropout: fraction of nodes in layer that are deactivated
    :param imagenet: boolean specifying whether or not to use pretrained imagenet weights in initialization model
    :return: multi-class averaged AUC on test set
    """

    # collect model and test generator
    model, test_generator = run_efficientnet(isic, blood, img_dir, label_dir, test_size, x_col, y_col, augment,
                                             validation_split, learning_rate, img_length, img_width, batch_size, epochs,
                                             color, dropout, imagenet)

    # compute OneVsRest multi-class macro AUC on the test set
    OneVsRest_auc = roc_auc_score(test_generator.classes, model.predict(test_generator), multi_class='ovr',
                                  average='macro')

    return OneVsRest_auc


AUC = test_evaluation(isic=False, blood=True,
                      img_dir="/Users/IrmavandenBrandt/Downloads/Internship/blood_data/9232-29380-bundle-archive/dataset2-master/dataset2-master/images",
                      label_dir=None, test_size=None, x_col="path", y_col="class",
                      augment=True, validation_split=0.2, learning_rate=0.000001, img_length=32, img_width=32,
                      batch_size=128, epochs=1, color=True, dropout=0.2, imagenet=True)
