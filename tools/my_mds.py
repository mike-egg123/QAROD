import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets,manifold

def plot_MDS(*data):
    '''
    绘制经过 使用 MDS 降维到二维之后的样本点
    '''
    X,y=data
    mds=manifold.MDS(n_components=2)
    #原始数据集转换到二维
    X_r=mds.fit_transform(X)

    ### 绘制二维图形
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    # 颜色集合，不同标记的样本染不同的颜色
    colors=((1,0,0),(0,1,0),(0,0,1),(0.5,0.5,0),(0,0.5,0.5),(0.5,0,0.5),(0.4,0.6,0),(0.6,0.4,0),(0,0.6,0.4),(0.5,0.3,0.2))
    for label ,color in zip( np.unique(y),colors):
        position=y==label
        ax.scatter(X_r[position,0],X_r[position,1],label="target= %d"%label,color=color)

    ax.set_xlabel("X[0]")
    ax.set_ylabel("X[1]")
    ax.legend(loc="best")
    ax.set_title("MDS")
    plt.show()

if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    print(X)
    print(y)
    mds = manifold.MDS(n_components=2)
    X_r = mds.fit_transform(X)
    print(X_r)
    plot_MDS(X, y)
