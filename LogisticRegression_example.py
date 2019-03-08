# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 16:16:44 2019

@author: nabhinav
"""
#Logistic Regression - Digit Recognition 
#Recognize the hand written digits
#there are 1797 digit images present and we  need to conver the images into some kind of a feature vector to make it usable.



import numpy as np 
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd

#Trainig set from scikitlearn , train LR on that and then predict 

from sklearn.datasets import load_digits

digits = load_digits()

dir(digits)
type(digits)

#each of the images are stored in the form of an array of numbers
digits.data[11]

digits.images[1]

#TO see the image that is given in the dataset. THe below command plots the image or shows the image in the data.

plt.gray()
plt.matshow(digits.images[11])

digits.target[100]
type(digits.target)

digits.target_names[0]

#fitting the model 
from sklearn.model_selection import train_test_split
X=digits.data
Y=digits.target

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25)

len(X_train)
len(Y_train)

#Importing the logistic Regression model 
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

#Fitting the LR model to our current training set

model.fit(X_train,Y_train)

model.score(X_test,Y_test)

plt.matshow(digits.images[211])
digits.data[211]

digits.target[211]

#Predicting random data points from our model
model.predict(digits.data[0:10])

#To get the predicted values from the test set
Y_predicted = model.predict(X_test)

#Evaluating the model by using a confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test,Y_predicted)
cm

#visualize confusion matrix
import seaborn as sn
plt.figure(figsize = (25,15))
sn.heatmap(cm,annot = True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
