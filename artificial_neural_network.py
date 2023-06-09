# -*- coding: utf-8 -*-
"""Artificial Neural Network.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hK6egbbhYt7fSkIOIwa52gMjHUi0a6Eo
"""

import numpy as p
import pandas as pd

dataset = pd.read_csv("/content/Churn_Modelling.csv")
print(dataset.head())
X = dataset.iloc[:,3:13]
Y = dataset.iloc[:,13]

print(X.head())

print(Y.head())

geography=pd.get_dummies(X["Geography"],drop_first=True)
gender=pd.get_dummies(X['Gender'],drop_first=True)

print(geography.head())

print(gender)

X=pd.concat([X,geography,gender],axis=1)
print(X.head())

X = X.drop(['Geography','Gender'],axis=1)
print(X.head())

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

print(x_train)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout

# initialising the ANN
classifier = Sequential()

# adding the input layer and the first hidden layer
classifier.add(Dense(units=6,kernel_initializer='he_uniform',activation='relu',input_dim=11))

# adding the second hidden layer
classifier.add(Dense(units=6,kernel_initializer='he_uniform',activation='relu'))

# adding the output layer
classifier.add(Dense(units=1,kernel_initializer='glorot_uniform',activation='sigmoid'))

# compiling the ANN
classifier.compile(optimizer='Adamax',loss='binary_crossentropy',metrics=['accuracy'])

# fitting the ANN to the training dataset
model_history = classifier.fit(x_train,y_train,validation_split=0.33,batch_size=10,epochs=100)

# list all data history
print(model_history.history.keys())

import matplotlib.pyplot as plt

plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model_history')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()

y_predict = classifier.predict(x_test)