from threading import main_thread
import banpei
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from pip import main
from scipy.stats import pearsonr
import numpy as np
from sklearn import manifold
matplotlib.use('TKAgg')

def sst_comparison(data1, data2):
    n1 = len(data1)
    model = banpei.SST(w=50)
    results1 = model.detect(data1)
    ax1 = plt.subplot(8, 1, 1)
    plt.sca(ax1)
    plt.plot([t for t in range(0, n1)], data1)
    ax2 = plt.subplot(8, 1, 3)
    plt.sca(ax2)
    plt.plot([t for t in range(0, n1)], results1)
    n2 = len(data2)
    ax3 = plt.subplot(8, 1, 5)
    plt.sca(ax3)
    plt.plot([t for t in range(0, n2)], data2)
    model = banpei.SST(w=50)
    results2 = model.detect(data2)
    ax4 = plt.subplot(8, 1, 7)
    plt.sca(ax4)
    plt.plot([t for t in range(0, n2)], results2)
    plt.show()

def pearson_comparison(data1, data2):
    return pearsonr(data1, data2)

if __name__ == '__main__':
    data1 = pd.read_csv('tools\periodic_waves.csv')['y'].to_list()
    data2 = [t for t in range(0, 530, 1)] + [-500] * 500 + [t for t in range(530, -410, -2)]
    data3 = data1[500:1000] + data1[0:500] + data1[500:1000]
    data4 = [t for t in range(120, -410, -1)] + [500] * 500 + [t for t in range(-410, 530, 2)]
    sst_comparison(data1, data3)
    r, p = pearson_comparison(data1, data3)
    print(r, p)

