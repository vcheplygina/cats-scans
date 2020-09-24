# %%
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

#
#
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

# learning rate used to be 0.000001
