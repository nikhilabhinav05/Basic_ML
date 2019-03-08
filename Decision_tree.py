# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 15:11:33 2019

@author: nabhinav
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn as sk

import os

pwd

cars = pd.read_csv('car_evaluation.csv')

cars.head(5)

# we need to predictthe car quality based on the features (price, maintainence, no. of doors , boot space , persons fitting  safety)

#So we need to first conver our feature var to numeric vals

cars.describe()

#converting the categorical var to numerical 
from sklearn.preprocessing import LabelEncoder

#seperating our feature var and target vars
cars_f=cars.drop('quality',axis =1)

cars_p = cars['quality']

cars_f.head()


cars.head()

type(cars_f.dtypes)

#Converting categories in Cars_f to numerical

cars_f.head()
#changed all col names to smaller and simpler once
cars_f = cars_f.rename(columns={'buying_price': 'bprice', 'MainTcost': 'mcost', 'number of doors':'door', 'number of person': 'person'})

le_bprice=LabelEncoder()
le_mcost=LabelEncoder()
le_door=LabelEncoder()

le_person=LabelEncoder()
le_boot=LabelEncoder()

le_safety=LabelEncoder()

#adding these columns to the DF , so replacing the categorical with numerical var

cars_f['bp']=le_bprice.fit_transform(cars_f['bprice'])


cars_f['mc']=le_bprice.fit_transform(cars_f['mcost'])


cars_f['d']=le_bprice.fit_transform(cars_f['door'])


cars_f['pax']=le_bprice.fit_transform(cars_f['person'])


cars_f['bs']=le_bprice.fit_transform(cars_f['boot'])


cars_f['saf']=le_bprice.fit_transform(cars_f['safety'])

#Now that we have all of our var in the numerical form , we will drop the respective categorical cols of these  features

cars_f.columns.get_loc("bp")

cars_f.drop(cars_f.columns[[0,1,2,3,4,5]], axis=1)

#converting our target var tonumeric form
 
 

type(cars_p)



cars.drop(cars.columns[[0,1,2,3,4,5,6]], axis=1)

cars.head()

cars_p = cars

type(cars_p)


cars_p = cars_p.rename(columns={'boot space': 'person'})

import pandas as pd

cars_p

type(cap)

ca=cars_p

cap = ca.drop(['bprice' , 'mcost' , 'door', 'person', 'safety'] , axis = 1)

#final target dataset
cap.head()

le_quality = LabelEncoder()

cap['q']=le_quality.fit_transform(cap['quality'])


cap['q'] = le_quality.fit_transform(cap['quality'])

cap.drop(['quality'], axis=1)


# cars_f is features ; cap is the target data set 

#splitting DS to Train and test 

from sklearn.model_selection import train_test_split

X = cars_f


Y= cap

Y= Y.drop(['quality'], axis=1)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25)

len(X_train)

X.head()

X_train.describe()
from sklearn import tree

model = tree.DecisionTreeClassifier()

model.fit(X_train, Y_train)

model.score(X_test, Y_test)

model.predict([[2,2,0,0,1,0]])

type(X)

type(Y)

X.head()

Y.head()