import os
import banpei
import pandas as pd
from sklearn import manifold
import numpy as np
import time

def read_data(filename):
    df = pd.read_csv(filename)
    return df

def get_filepaths(parent):
    filepaths = []
    for filepath, dirnames, filenames in os.walk(parent):
        for filename in filenames:
            filepaths.append(os.path.join(filepath, filename))
    return filepaths

def get_sst_score(values):
    for i in range(0, len(values)):
        values[i] = float(values[i])
    model = banpei.SST(w=25)
    results = model.detect(values)
    return results

if __name__ == '__main__':
    # filepaths = get_filepaths(r"E:\文档\课程\大四\毕设\程序\data\banpei")
    # n = len(filepaths)
    # i = 0
    # for filename in filepaths:
    #     df = read_data(filename)
    #     values = df["value"].values.tolist()
    #     model = banpei.SST(w=60)
    #     results = model.detect(values)
    #     for j in range(len(results)):
    #         if results[j] < 0.0001:
    #             results[j] = 0
    #     df["anomaly_score"] = results
    #     df.to_csv(filename)
    #     i += 1
    #     print(filename + "已完成异常检测，进度为%" + str(i / n * 100))
    filename = "../data/qar/148923-20160101.csv"
    df = read_data(filename)
    values1 = df["N11"].values.tolist()[3:]
    sst1 = get_sst_score(values1)
    values2 = df["N12"].values.tolist()[3:]
    sst2 = get_sst_score(values2)
    values3 = df["VIB_N12"].values.tolist()[3:]
    sst3 = get_sst_score(values3)
    values4 = df["GSC"].values.tolist()[3:]
    sst4 = get_sst_score(values4)
    values5 = df["RALTC"].values.tolist()[3:]
    sst5 = get_sst_score(values5)
    values6 = df["IAS"].values.tolist()[3:]
    sst6 = get_sst_score(values6)
    ssts = np.stack((sst1, sst2, sst3, sst4, sst5, sst6), 0)
    mds = manifold.MDS(n_components=2)
    X_r = mds.fit_transform(ssts)
    print(X_r)



