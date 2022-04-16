from pyod.models.knn import KNN   # kNN detector
from pyod.models.lof import LOF
from pyod.utils import generate_data, standardizer
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize
from pyod.models.combination import aom, moa, average, maximization, median
import numpy as np

contamination = 0.1  # percentage of outliers
n_train = 200  # number of training points
n_test = 100  # number of testing points

X_train, y_train, X_test, y_test = generate_data(
    n_train=n_train, n_test=n_test, contamination=contamination, n_features=1)  # 维数越高，预测效果越差

print(X_train)

# train kNN detector
clf_name = 'LOF'
clf = LOF()
clf.fit(X_train)

# get the prediction labels and outlier scores of the training data
y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
y_train_scores = clf.decision_scores_  # raw outlier scores

# get the prediction on the test data
y_test_pred= clf.predict(X_test)  # outlier labels (0 or 1)
y_test_scores = clf.decision_function(X_test)  # outlier scores

# it is possible to get the prediction confidence as well
# y_test_pred, y_test_pred_confidence = clf.predict(X_test, return_confidence=True)  # outlier labels (0 or 1) and confidence in the range of [0,1]

# evaluate and print the results
print("\nOn Training Data:")
evaluate_print(clf_name, y_train, y_train_scores)
print("\nOn Test Data:")
evaluate_print(clf_name, y_test, y_test_scores)

# visualize(clf_name, X_train, y_train, X_test, y_test, y_train_pred,
#           y_test_pred, show_figure=True, save_figure=False)

# # Model Combination
# from pyod.models.knn import KNN   # kNN detector
# from pyod.utils import generate_data, standardizer
# from pyod.utils.data import evaluate_print
# from pyod.models.combination import aom, moa, average, maximization, median
# import numpy as np
# from sklearn.model_selection import train_test_split
#
# X, y= generate_data(train_only=True)  # load data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
#
# # initialize 20 base detectors for combination
# k_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140,
#             150, 160, 170, 180, 190, 200]
# n_clf = len(k_list) # Number of classifiers being trained
#
# train_scores = np.zeros([X_train.shape[0], n_clf])
# test_scores = np.zeros([X_test.shape[0], n_clf])
#
# # standardizing data for processing
# X_train_norm, X_test_norm = standardizer(X_train, X_test)
#
# for i in range(n_clf):
#     k = k_list[i]
#
#     clf = KNN(n_neighbors=k, method='largest')
#     clf.fit(X_train_norm)
#
#     train_scores[:, i] = clf.decision_scores_
#     test_scores[:, i] = clf.decision_function(X_test_norm)
#
# # scores have to be normalized before combination
# train_scores_norm, test_scores_norm = standardizer(train_scores, test_scores)
#
# # Combination by average
# y_by_average = average(test_scores_norm)
# evaluate_print('Combination by Average', y_test, y_by_average)
#
# # Combination by max
# y_by_maximization = maximization(test_scores_norm)
# evaluate_print('Combination by Maximization', y_test, y_by_maximization)
#
# # Combination by median
# y_by_median = median(test_scores_norm)
# evaluate_print('Combination by Median', y_test, y_by_median)
#
# # Combination by aom
# y_by_aom = aom(test_scores_norm, n_buckets=5)
# evaluate_print('Combination by AOM', y_test, y_by_aom)
#
# # Combination by moa
# y_by_moa = moa(test_scores_norm, n_buckets=5)
# evaluate_print('Combination by MOA', y_test, y_by_moa)
#
# # SUOD
# from pyod.models.copod import COPOD
# from pyod.models.iforest import IForest
# from pyod.models.lof import LOF
# from pyod.models.suod import SUOD
#
#
# # initialized a group of outlier detectors for acceleration
# # detector_list = []
# # for i in range(n_clf):
# #     k = k_list[i]
# #     detector_list.append(KNN(n_neighbors=k, method="largest"))
#
# detector_list = [KNN(n_neighbors=15), LOF(n_neighbors=20),
#                  LOF(n_neighbors=25), LOF(n_neighbors=35),
#                  COPOD(), IForest(n_estimators=100),
#                  IForest(n_estimators=200)]
#
# # decide the number of parallel process, and the combination method
# # then clf can be used as any outlier detection model
# clf = SUOD(base_estimators=detector_list, n_jobs=2, combination='average',
#            verbose=False)
# clf.fit(X_train_norm)
# test_scores = clf.decision_function(X_test_norm)
# evaluate_print("SUOD", y_test, test_scores)


