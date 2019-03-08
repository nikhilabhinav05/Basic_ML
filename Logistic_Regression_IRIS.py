# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 17:56:53 2019

@author: nabhinav
"""

from sklearn.datasets import load_iris
iris = load_iris()

dir(iris)
type(iris)

iris

df = pd.DataFrame(iris.data, columns=iris.feature_names)

df.head(5)

df['target'] = iris.target

df['flower_name']=df.target.apply(lambda x:iris.target_names[x])

df0=df[df.target==0]
df1=df[df.target==1]
df2=df[df.target==2]

df0.head(5)
df1.head(5)
df2.head(5)

plt.figure(figsize=(20,15))
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'], color='blue',marker='+', s=45 )
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'], color='red',marker='.',s=45)
plt.scatter(df2['sepal length (cm)'],df2['sepal width (cm)'], color='orange',marker='*',s=45)


from sklearn.model_selection import train_test_split

X= df.drop(['target','flower_name'],axis = 'columns')

Y=df.target

X.head()

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.35)

from sklearn.linear_model import LogisticRegression

model=LogisticRegression()

model.fit(X_train,Y_train)

model.score(X_test,Y_test)

df
X

#Confusion matrix 
from sklearn.metrics import confusion_matrix

Y_predicted = model.predict(X_test)
cm = confusion_matrix(Y_predicted,Y_test)
cm

plt.imshow(cm, cmap='binary')

import seaborn as sn 
plt.figure(figsize=(10,5))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylablel('Actual')

from sklearn.feature_selection import chi2

scores,pvalues=chi2(X,Y)

pvalues

scores

X.head(5)
type(X)

list(X1)

X1=X.drop(['sepal width (cm)'], axis='columns')

X1_train,X1_test,Y_train,Y_test = train_test_split(X1,Y, test_size=0.35)

model.fit(X1_train,Y_train)

model.score(X1_test,Y_test)
scores,pvalues=chi2(X1,Y)
