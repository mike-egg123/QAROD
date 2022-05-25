# Import libraries, core libraries are numpy and sklearn
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
import os
os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'

# 融合数据集（仅第一次使用）
def data_merge():
    # Load data set
    folder = r'E:\文档\课程\大四\实验室\QAR\AutoEncoderData\archive\1st_test'
    data_dir = folder+r'\1st_test'
    merged_data = pd.DataFrame()

    for filename in os.listdir(data_dir):
    #     print(filename)
        dataset = pd.read_csv(os.path.join(data_dir, filename), sep='\t')
        dataset_mean_abs = np.array(dataset.abs().mean())
        dataset_mean_abs = pd.DataFrame(dataset_mean_abs.reshape(1,8))
        dataset_mean_abs.index = [filename]
        merged_data = merged_data.append(dataset_mean_abs)
    merged_data.reset_index(inplace=True)  # reset index to get datetime as columns

    # 将原始数据集聚合为单个csv文件
    merged_data.columns = ['Datetime','Bearing 1l','Bearing 1r', 'Bearing 2l', 'Bearing 2r', 'Bearing 3l', 'Bearing 3r', 'Bearing 4l', 'Bearing 4r'] # rename columns
    merged_data.sort_values(by='Datetime',inplace=True)
    merged_data.to_csv('1st_test_resmaple_10minutes.csv')

# 读取数据集
def data_load():
    merged_data = pd.read_csv('2nd_test_resmaple_10minutes.csv',index_col='Datetime',usecols=['Datetime','Bearing 1','Bearing 2','Bearing 3','Bearing 4'])
    merged_data.index = pd.to_datetime(merged_data.index, format='%Y.%m.%d.%H.%M.%S')
    print(merged_data.head())
    # merged_data.plot()

    # Data pre-processing
    # Split the training and test sets using 2004-02-13 23:52:39 as the cut point
    dataset_train = merged_data[:'2004-02-13 23:52:39']
    dataset_test  = merged_data['2004-02-13 23:52:39':]
    # dataset_train.plot(figsize = (12,6))

    """
    Normalize data
    """
    scaler = preprocessing.MinMaxScaler() # 归一化

    X_train = pd.DataFrame(scaler.fit_transform(dataset_train), # Find the mean and standard deviation of X_train and apply them to X_train
                                  columns=dataset_train.columns,
                                  index=dataset_train.index)

    # Random shuffle training data
    X_train.sample(frac=1)

    X_test = pd.DataFrame(scaler.transform(dataset_test),
                                 columns=dataset_test.columns,
                                 index=dataset_test.index)
    return X_train,X_test


# 构建自编码器模型
def AutoEncoder_build(X_train, act_func):
    tf.random.set_seed(10)

    # act_func = 'elu'

    # Input layer:
    model = tf.keras.Sequential()  # Sequential() is a container that describes the network structure of the neural network, sequentially processing the model

    # First hidden layer, connected to input vector X.
    model.add(tf.keras.layers.Dense(10, activation=act_func,  # activation function
                                    kernel_initializer='glorot_uniform',  # Weight initialization
                                    kernel_regularizer=tf.keras.regularizers.l2(0.0),
                                    # Regularization to prevent overfitting
                                    input_shape=(X_train.shape[1],)
                                    )
              )

    model.add(tf.keras.layers.Dense(2, activation=act_func,
                                    kernel_initializer='glorot_uniform'))

    model.add(tf.keras.layers.Dense(10, activation=act_func,
                                    kernel_initializer='glorot_uniform'))

    model.add(tf.keras.layers.Dense(X_train.shape[1],
                                    kernel_initializer='glorot_uniform'))

    model.compile(loss='mse', optimizer='adam')  # 设置编译器

    print(model.summary())
    # tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, rankdir="LR",)

    return model

# 开训
def AutoEncoder_main(model,Epochs,BATCH_SIZE,validation_split):
    # Train model for 100 epochs, batch size of 10:
	# noise
    # 向训练集的输入中加入随机噪声，这是DAE！
    factor = 0.5
    X_train_noise = X_train + factor * np.random.normal(0,1,X_train.shape)
    X_train_noise = np.clip(X_train_noise,0.,1.)
    
    history=model.fit(np.array(X_train_noise),np.array(X_train),
                      batch_size=BATCH_SIZE,
                      epochs=Epochs,
                      shuffle=True,
                      validation_split=validation_split, # Training set ratio
#                       validation_data=(X_train,X_train), # Validation set
                      verbose = 1)

    return history

# Figure 展示训练过程的mse，发现很快收敛，且较稳定
def plot_AE_history(history):
    plt.plot(history.history['loss'],
             'b',
             label='Training loss')
    plt.plot(history.history['val_loss'],
             'r',
             label='Validation loss')
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Loss, [mse]')
    plt.ylim([0,.1])
    plt.show()

# 测试
def test_gpu():
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")

# 查看模型的认知水平，通过还原原始数据，计算真实原始数据和重构数据的误差，得到阈值
def get_threshold():
    X_pred = model.predict(np.array(X_train))
    X_pred = pd.DataFrame(X_pred,
                        columns=X_train.columns)
    X_pred.index = X_train.index

    scored = pd.DataFrame(index=X_train.index)
    scored['Loss_mae'] = np.mean(np.abs(X_pred-X_train), axis = 1)
    sns.displot(scored['Loss_mae'],
                bins = 10,
                kde= True,
                color = 'blue')
    plt.xlim([0.0,.5])
    plt.show()

def get_result():
    X_pred = model.predict(np.array(X_test))
    X_pred = pd.DataFrame(X_pred,
                        columns=X_test.columns)
    X_pred.index = X_test.index


    threshold = 0.3
    scored = pd.DataFrame(index=X_test.index)
    scored['Loss_mae'] = np.mean(np.abs(X_pred-X_test), axis = 1)
    scored['Threshold'] = threshold
    scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
    print(scored.head())

    X_pred_train = model.predict(np.array(X_train))
    X_pred_train = pd.DataFrame(X_pred_train,
                        columns=X_train.columns)
    X_pred_train.index = X_train.index

    scored_train = pd.DataFrame(index=X_train.index)
    scored_train['Loss_mae'] = np.mean(np.abs(X_pred_train-X_train), axis = 1)
    scored_train['Threshold'] = threshold
    scored_train['Anomaly'] = scored_train['Loss_mae'] > scored_train['Threshold']
    scored = pd.concat([scored_train, scored])

    scored.plot(logy=True,  figsize = (10,6), ylim = [1e-2,1e2], color = ['blue','red'])
    plt.show()

if __name__ == '__main__':
    # data_merge()
    X_train, X_test = data_load()
    model = AutoEncoder_build(X_train, 'elu')
    history = AutoEncoder_main(model=model,Epochs=100,BATCH_SIZE=10,validation_split=0.05)
    # plot_AE_history(history)
    # get_threshold()
    get_result()

    

    

    

