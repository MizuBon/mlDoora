# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 15:35:51 2021

@author: BonBon
"""

import sys
import scipy
import numpy
import matplotlib
import pandas
import pandas as pd
import sklearn

from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#Loading the Dataset one moment
path = r"C:\Users\BonBon\Downloads\safeDoors.csv"
names = ['Program_Name', 'Code']
safeData = read_csv(path, names=names)

path = "https://raw.githubusercontent.com/MizuBon/mlDoora/main/backdoors.csv"
#names = ['Program_Name', 'Code']
backData = read_csv(path, names=names)

#Now we gonna merge bois lets go
overallData = pd.concat([safeData,backData], ignore_index=True)
print(overallData.head(60))

#Lets look at our data a bit
print(safeData.shape)
print(safeData.describe())
print(safeData)

print(backData.shape)
print(backData.describe())
print(backData)

#Now we're gonna actually split the data a bit
# Split-out validation dataset
array = overallData.values
X = array[:,0:1]
y = array[:,1]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=2, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))