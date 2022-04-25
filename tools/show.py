import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

def show_pic(x, y):
    plt.plot(x, y)
    plt.show()
    return

# 双y轴
def show_pic_double_y(x, y1, y2):
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.plot(x, y1, 'r', label="values")
    ax1.set_ylabel('values')
    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(x, y2, 'g', label="anomaly_score")
    ax2.set_ylabel('anomaly_score')
    plt.show()
    return

def read_data(filename):
    df = pd.read_csv(filename)
    return df

# 肘部法寻找最佳的k
def elbow_best_k(values):
    sse = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k).fit(np.array(values).reshape(-1, 1))
        sse.append(km.inertia_)
    diff = []
    for i in range(1, len(sse)):
        diff.append(sse[i - 1] - sse[i])
    labels = KMeans(n_clusters=2).fit_predict(np.array(diff).reshape(-1, 1))
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            return i + 2, sse, diff, labels
    return 3, sse, diff, labels

if __name__ == '__main__':
    matplotlib.use('TKAgg')
    df = read_data("data/qar/148923-20160101.csv")
    drift = df["DRIFT"].values.tolist()[3:]
    plt.plot(range(0, len(drift)), drift, linewidth=0.7)
    plt.show()
    # k, sse, diff, labels = elbow_best_k(N12)
    # ax1 = plt.subplot(3, 1, 1)
    # plt.sca(ax1)
    # plt.plot(range(2, 11), sse)
    # plt.scatter(k, sse[k - 2])
    # ax2 = plt.subplot(3, 1, 2)
    # plt.sca(ax2)
    # plt.plot(range(2, 10), diff)
    # ax3 = plt.subplot(3, 1, 3)
    # plt.sca(ax3)
    # plt.plot(range(2, 10), labels)
    # plt.show()
    # x = range(0, len(N12))
    # show_pic(x, N12)
    # show_pic_double_y(x, N11, anomaly_score)