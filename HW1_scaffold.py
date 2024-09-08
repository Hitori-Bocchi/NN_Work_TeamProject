import random
random.seed(112358)

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# TensorFlow and tf.keras
import tensorflow as tf

%matplotlib inline
# 1.1
# your code here 

# 1.1.2
# your code here

# 1.2
# build your NN 
# your code here


# compile it and run it
# your code here 


# plot train and val acc as  a function of epochs
# your code here


# primer to print: 
# print("NN_model_train_auc:", roc_auc_score(y_train, y_hat))
# your code here 


# your code here

# 1.3
# Fit the logistic regression model
# your code here

# 1.4
# 1.4.1
# your code here

# 1.4.1要写解释
# 1.4.2
# your code here

# 1.4.3
# your code here

# 1.4.4
# your code here

# 1.4.4要写解释

# 1.5
def progressbar(n_step, n_total):
    """Prints self-updating progress bar to stdout to track for-loop progress
    
    There are entire 3rd-party libraries dedicated to custom progress-bars.
    A simple function like this is often more than enough to get the job done.
    
    :param n_total: total number of expected for-loop iterations
    :type n_total: int
    :param n_step: current iteration number, starting at 0
    :type n_step: int

    .. example::
    
        for i in range(n_iterations):
            progressbar(i, n_iterations)
            
    .. source:
    
        This function is a simplified version of code found here:
        https://stackoverflow.com/questions/3160699/python-progress-bar/15860757#15860757
    """
    n_step = n_step + 1
    barlen = 50
    progress = n_step / n_total
    block = int(round(barlen * progress))
    status = ""
    if n_step == n_total:
        status = "Done...\r\n\n"
    text = "\r [{0}] {1}/{2} {3}".format(
        "=" * block + "-" * (barlen - block),
        n_step,
        n_total,
        status,
    )
    sys.stdout.write(text)
    sys.stdout.flush()

%%time
# Bootstrap and train your networks and get predictions on fixed X test
# your code here


# generate your plot
# your code here

# 1.5此处要写解释
# 1.6
# your code here

# 1.6此处要写解释

# 2.1
# your code here 
train_data = pd.read_csv('kmnist_train.csv')    #这个是panda的dataframe，我直接是把文件放在本地跑的，和ed上面的目录不一样
#上传的时候我会修改相对路径

images = train_data.iloc[:, :-1].values     #提取图像数据和标签
labels = train_data.iloc[:, -1].values      #其实这里不确定，因为他说的是用_kmnist训练数据，但是又说
                                            #utilize kmnist_test.csv in question 2.3.4 only  所以我暂时当作他是对的

image_0 = images[labels == 0][0].reshape(28, 28)    # 这里是选择第一个0和第一个1的样本进行可视化
image_1 = images[labels == 1][0].reshape(28, 28)    #我测试过，会出现模糊的0和1

plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.title("Handwritten 0")
plt.imshow(image_0, cmap='gray')
plt.subplot(1, 2, 2)
plt.title("Handwritten 1")
plt.imshow(image_1, cmap='gray')
plt.show()
# 2.2
# your code here

# 2.2此处要写解释
X1_train, X1_val, Y1_train, Y1_val = train_test_split(images, labels, test_size=0.3, random_state=42)

#这里就是定义的模型
model_overfit = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])    #跑了一下，这里很早就过拟合了，差不多100个epoch就过拟合了

model_overfit.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# 2.3
#your code here
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=100)    #在这里我用的早停
model = tf.keras.Sequential([    #我这里使用了早停、Dropout和L2正则化
#实际上这个模型在
    tf.keras.layers.Dense(100, activation='relu', input_shape=(784,), kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(100, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(100, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X1_train,Y1_train,epochs=2000, batch_size=512, validation_data=(X1_val, Y1_val),callbacks=[callback])

# 2.3.1
# your code here
model_overfit.summary()
model.summary()
# 2.3.2
# your code here

# 2.3.3
# your code here
plt.figure(figsize=(12,5))    #他说要绘制验证集和训练集的测试准确率，我把两个模型的图都放在这里了
plt.subplot(1, 2, 1)
plt.plot(history_overfit.history['accuracy'], label='Training Accuracy')
plt.plot(history_overfit.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Overfitting Example: Training vs Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Regularized Model: Training vs Validation Accuracy')
plt.show()
# 2.3.4
# your code here
