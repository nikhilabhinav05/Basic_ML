# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 13:07:59 2019

@author: nabhinav
"""

import numpy as np 
import pandas as pd
import sklearn as sk
import matplotlib as plt


from sklearn.datasets import *
iris = load_iris()

iris.feature_names

type(iris)
dir(iris)

#converting the function to a data frame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.head(10)

#adding the target column to this data frame
df['Target']=iris.target
df.head(5)

dir(iris)
#Assigns names to flowers based on the target value present in the dataframe
df['flower_name']=df.Target.apply(lambda x:iris.target_names[x])

df['flower_name']=iris.target_names

from matplotlib import pyplot as plt
#seperating the three classes of flowers into 3 different types of df
df0=df[df.Target==0]
df1=df[df.Target==1]
df2=df[df.Target==2]

df2.head(5)

#scatter plot
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'], color='blue',marker='+')

plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'], color='red',marker='.')

#creating a new DF by Removing the Y (dependent column) from the data frame 
X=df.drop(['Target','flower_name'], axis='columns')
X.head(5)

# Creating Y(dependent variable) Adding Y as the target for the training data
Y=df.Target
Y.head(5)

#fitting the model 
from sklearn.model_selection import train_test_split


#splitting the data into train_test 
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

len(X_train)

#importing SVM
from sklearn.svm import SVC

model = SVC()

#Training the model
model.fit(X_train,Y_train)

#evaluates the score of the model
model.score(X_test,Y_test)

#Increasing the C (regularization value)
model =SVC(C=10)

#increasing regularization is decreasing the score

help(sk)

