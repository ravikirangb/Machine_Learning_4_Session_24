# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 17:45:43 2018

@author: 1000091


You use only Pclass, Sex, Age, SibSp (Siblings aboard), Parch (Parents/children aboard),
and Fare to predict whether a passenger survived.

************************************************
************************************************
My Jupyter notebook has some problem- Matplotlib works on Spyder but not on Jupyter- I checked version pointer, uninstalled 
Pyqt, Matplotlib and Seaborn- still the same issue.- PLEASE GUIDE

************************************************
************************************************
PLEASE GUIDE

************************************************
************************************************
"""
print ("**************** My Jupyter notebook has some problem- Matplotlib works on Spyder but not on Jupyter-\
       I checked version pointer, uninstalled  \
       Pyqt, Matplotlib and Seaborn- still the same issue.- \
       **********************************************PLEASE GUIDE********************************************")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn import tree
from sklearn.metrics import classification_report


# Read the data
 
titanic = pd.read_csv("titanic.csv")
print(titanic.head())

grouping_1 = titanic.groupby(['Pclass', 'Sex']).mean()
print (grouping_1['Survived'].plot.bar())
plt.show()

titanic = titanic[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]
titanic.isnull().any()

print(titanic.head())

titanic["Age"].fillna(titanic["Age"].mean(),inplace=True)
titanic.Sex.value_counts()

label_encoder = preprocessing.LabelEncoder()

titanic['Sex'] = label_encoder.fit_transform(titanic["Sex"])

print (titanic['Sex'])

X = titanic[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

y = titanic['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

model1 = DecisionTreeClassifier(random_state=0)
model1.fit(X_train, y_train)
predicted = model1.predict(X_test)
print("test data set accuracy \n")
print (format(metrics.accuracy_score(y_test, predicted) * 100,'.2f'), '%.')

Xvscores = cross_val_score(DecisionTreeClassifier(random_state=0), X, y, cv=20)
print("Xvscores and Xvscores mean:-- \n",Xvscores, Xvscores.mean())

Xpredictions = cross_val_predict(DecisionTreeClassifier(random_state=0), X, y, cv=15)
print("Accuracy for test data set:\n")
print (format(metrics.accuracy_score(y_test, predicted) * 100,'.3f'), '% - percentage')

params= [{'max_depth': range(3, 7)}]

model2 = GridSearchCV(DecisionTreeClassifier(random_state=0), params)
model2.fit(X_train, y_train)

print("Best parameters found on development set:\n")
print(model2.best_params_, '\n')

print("Accuracy for test data set:\n")
predicted = model2.predict(X_test)
print (format(metrics.accuracy_score(y_test, predicted) * 100,'.3f'), '% - percentage')

print (model2.predict_proba(np.array([[2, 1, 3, 0, 2, 0.0]])))
