# %%
from sklearn.metrics import roc_auc_score
from TF_generators_models import run_model
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve


def test_evaluation(isic, blood, img_dir, label_dir, test_size, x_col, y_col, augment, learning_rate,
                    img_length, img_width, batch_size, epochs, color, dropout, imagenet, model_choice):
    """
    :param isic: boolean specifying whether data needed is ISIC data or not
    :param blood: boolean specifying whether data needed is blood data or not
    :param img_dir: directory where images are found
    :param label_dir: directory where labels are found
    :param test_size: split value used to split part of dataframe into test set
    :param x_col: column in dataframe containing the image paths
    :param y_col: column in dataframe containing the target labels
    :param augment: boolean specifying whether to use data augmentation or not
    :param img_length: target length of image in pixels
    :param img_width: target width of image in pixels
    :param batch_size: amount of images processed per batch
    :param learning_rate: learning rate used by optimizer
    :param epochs: number of epochs the model needs to run
    :param color: boolean specifying whether the images are in color or not
    :param dropout: fraction of nodes in layer that are deactivated
    :param imagenet: boolean specifying whether or not to use pretrained imagenet weights in initialization model
    :param model_choice: model architecture to use for convolutional base (i.e. resnet or efficientnet)
    :return: multi-class averaged AUC on test set
    """

    # collect model and test generator
    model, test_generator = run_model(isic, blood, img_dir, label_dir, test_size, x_col, y_col, augment,
                                             learning_rate, img_length, img_width, batch_size, epochs,
                                             color, dropout, imagenet, model_choice)

    # compute OneVsRest multi-class macro AUC on the test set
    OneVsRest_auc = roc_auc_score(test_generator.classes, model.predict(test_generator), multi_class='ovr',
                                  average='macro')

    # # compute individual auc scores for every class (OneVsRest)
    # if individual:
    #
    #     clf = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True)).fit(X, y)
    #
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
    #
    #     for i in range(n_class):
    #         fpr[i], tpr[i], thresh[i] = roc_curve(y_test, pred_prob[:, i], pos_label=i)

    return OneVsRest_auc


AUC = test_evaluation(isic=True, blood=False,
                      # img_dir="/Users/IrmavandenBrandt/Downloads/Internship/blood_data/9232-29380-bundle-archive"
                      #         "/dataset2-master/dataset2-master/images",
                      img_dir="/Users/IrmavandenBrandt/Downloads/Internship/ISIC2018/ISIC2018_Task3_Training_Input",
                      label_dir="/Users/IrmavandenBrandt/Downloads/Internship/ISIC2018"
                                "/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv",
                      # label_dir=None,
                      test_size=None, x_col="path", y_col="class",
                      augment=True, learning_rate=0.00001, img_length=60, img_width=80,
                      batch_size=128, epochs=20, color=True, dropout=0.2, imagenet=True, model_choice="resnet")

# learning rate used to be 0.000001
