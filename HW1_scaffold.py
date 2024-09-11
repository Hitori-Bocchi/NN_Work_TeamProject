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

# preprocessing
from sklearn import preprocessing 

# TensorFlow and tf.keras
import tensorflow as tf

# Some modules added
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

%matplotlib inline
# 1.1
# your code here 
# Read the dataset
df = pd.read_csv("flights.csv")
# Delete the value of airport is digit (Noise)
rows_to_drop = df[df['ORIGIN_AIRPORT'].str.isdigit() & df['DESTINATION_AIRPORT'].str.isdigit()].index
df.drop(rows_to_drop, inplace=True)

# Create a variable DELAY_OR_NOT that denotes whether ARRIVAL_DELAY is greater than or equal to 15 minutes
# if delayed, 1; else, 0
df['DELAY_OR_NOT'] = np.where(df['ARRIVAL_DELAY'] >= 15, 1, 0)

# 1.1.2
# your code here
# preprocess the data
# Having checked the data by hand, there is no missing value
categorical_features = ['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']

# one-hot-encode the non-numeric categorical variables
enc = preprocessing.OneHotEncoder(sparse=False)
encoded_features = enc.fit_transform(df[categorical_features])

# change the result to dataFrame. easy to check and concat
# Notice: old version sklearn using get_feature_names function
columns = enc.get_feature_names(categorical_features)
encoded_df_part = pd.DataFrame(encoded_features, columns=enc.get_feature_names(categorical_features))
df.drop(columns=categorical_features, inplace=True)

# Make sure index alignment
df = df.reset_index(drop=True)
encoded_df_part = encoded_df_part.reset_index(drop=True)

df = pd.concat([df, encoded_df_part], axis=1)

# Get predictors and response
predictors = df.drop(columns=['DELAY_OR_NOT'])
response = df['DELAY_OR_NOT']

# split the data into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(predictors, response, test_size=0.2, random_state=111)

# scale the data and print
# Do not scale y because values of y are 0 and 1
X_train_scaled = StandardScaler().fit_transform(X_train)
X_test_scaled = StandardScaler().fit_transform(X_test)
print(f"X_train shape: {X_train_scaled.shape}")
print(f"X_test shape: {X_test_scaled.shape}")
print(f"y_train : {y_train.shape}")
print(f"y_test : {y_test.shape}")

# 1.2
# build your NN 
# your code here
NN_model = Sequential(name="NN_model")
NN_model.add(Dense(200, activation="relu", kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001),
                   input_shape=(X_train_scaled.shape[1],)))
NN_model.add(Dropout(0.2))
NN_model.add(Dense(200,kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001), activation="relu"))
NN_model.add(Dropout(0.2))
NN_model.add(Dense(1, activation="sigmoid"))


# compile it and run it
# your code here 
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
NN_model.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
history = NN_model.fit(X_train_scaled, y_train, batch_size=64, epochs=50, validation_split=0.2,
                       callbacks=[early_stopping])

# plot train and val acc as  a function of epochs
# your code here
plt.title('Model Accuracy')
plt.plot(history.history['accuracy'], label='Train Accuracy', color='#FF9A98')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#75B594')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

# primer to print: 
# print("NN_model_train_auc:", roc_auc_score(y_train, y_hat))
# your code here 
y_train_predict_pro = NN_model.predict(X_train_scaled)
train_auc = roc_auc_score(y_train, y_train_predict_pro)
print(f"NN_model_train_auc: {train_auc:.4f}")

# your code here

# 1.3
# Fit the logistic regression model
# your code here
# replace the actual response values with the predicted
# values generated by the fitted NN_model
proxy_X_train = X_train_scaled
# pro > 0, the result is 'yes', else 'no'
proxy_y_train = np.where(NN_model.predict(X_train_scaled) > 0.5, 1, 0)
# original proxy_y_train is two-dimensional
proxy_y_train = proxy_y_train.reshape(-1)

# build the model and print the test accuracy
logreg = LogisticRegression(penalty='l2', C=100, solver='lbfgs', max_iter=1000)
logreg.fit(proxy_X_train, proxy_y_train)
proxy_y_test_pred = logreg.predict(X_test_scaled)
proxy_test_accuracy = accuracy_score(y_test, proxy_y_test_pred)
print(f"Logistic Regression Test Accuracy: {proxy_test_accuracy:.4f}")
# The result of local environment is
# NN_model_train_auc: 0.9994
# Logistic Regression Test Accuracy: 0.9802

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
model_overfit = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])    #跑了一下，这里很早就过拟合了，差不多100个epoch就过拟合了

model_overfit.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_overfit = model_overfit.fit(X1_train, Y1_train, epochs=2000, batch_size=512, validation_data=(X1_val, Y1_val))

model_overfit = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])    #跑了一下，这里很早就过拟合了，差不多100个epoch就过拟合了

model_overfit.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_overfit = model_overfit.fit(X1_train, Y1_train, epochs=2000, batch_size=512, validation_data=(X1_val, Y1_val))

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=200,restore_best_weights=True, verbose=1)    #在这里我用的早停
model = tf.keras.Sequential([    #我这里使用了早停、Dropout和L2正则化
#实际上这个模型在
    tf.keras.layers.Dense(100, activation='relu', input_shape=(784,), kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(100, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(100, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X1_train,Y1_train,epochs=2000, batch_size=512, validation_data=(X1_val, Y1_val),callbacks=[callback])
# 2.2此处要写解释

# 2.3.1
# your code here
model_overfit.summary()
model.summary()

# 2.3.2
# your code here
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]

print("Difference between training and validation accuracy:", final_train_acc - final_val_acc)
print("Difference between training and validation loss:", final_train_loss - final_val_loss)

# 2.3.3
# your code here
plt.figure(figsize=(12,5))
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
# 这里用测试集测试
test_data = pd.read_csv('kmnist_test.csv')
x_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test accuracy:", test_accuracy)
