import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score
import geohash
from sklearn.metrics import mean_squared_error
import random
from datetime import datetime
from sklearn import linear_model
import sklearn.cross_validation as cv
from sklearn.tree import DecisionTreeClassifier

input=pd.read_csv("D:/Machine Learning/NYCTraffic/file1.csv")
input.Pickuptime=pd.to_datetime(input.Pickuptime)
input['Hour']=input['Pickuptime'].apply(lambda x: x.hour)
input['Tip']=0
a=(input.Tip_amount >= 0.15* input.Fare_amount)
input['Tip'][a]=1
a=(input.Tip_amount < 0.15* input.Fare_amount)
input['Tip'][a]=0
input=input.round({'Zone' : 0})

target=input[['Tip']]
data=input[[col for col in input.columns if col not in ['Tip']]]
x_train, x_test, y_train, y_test = cv.train_test_split(data, target, test_size=2.0/10, random_state=0)
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(x_train, y_train)
logreg.predict(x_test)
'training R^2: %.2f',logreg.score(x_train, y_train)
'testing R^2: %.2f',logreg.score(x_test, y_test)

'training RMSE:', mean_squared_error(y_train, logreg.predict(x_train))
'testing RMSE:', mean_squared_error(y_test, logreg.predict(x_test))

##Using DecisionTree Classifier

clf = DecisionTreeClassifier(random_state=0, max_depth=20)
clf.fit(x_train, y_train)
clf.predict(x_test)
'training R^2: %.2f',clf.score(x_train, y_train)
'testing R^2: %.2f',clf.score(x_test, y_test)

'training RMSE:', mean_squared_error(y_train, clf.predict(x_train))
'testing RMSE:', mean_squared_error(y_test, clf.predict(x_test))

##Creating dummy variables for zones, houroftheday, and dayoftheweek, passengercount

features = pd.concat([pd.get_dummies(input.Zone, prefix='Zone'), pd.get_dummies(input.Hour, prefix='Hour'), pd.get_dummies(input.Weekday, prefix='Weekday'), pd.get_dummies(input.Passenger_count, prefix='Passenger_count')], axis=1)
input1=pd.concat([input, features], axis=1)

target=input1[['Tip']]
data=input1[[col for col in input1.columns if col not in ['Tip']]]
x_train, x_test, y_train, y_test = cv.train_test_split(data, target, test_size=2.0/10, random_state=0)
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(x_train, y_train)
logreg.predict(x_test)
'training R^2: %.2f',logreg.score(x_train, y_train)
'testing R^2: %.2f',logreg.score(x_test, y_test)

'training RMSE:', mean_squared_error(y_train, logreg.predict(x_train))
'testing RMSE:', mean_squared_error(y_test, logreg.predict(x_test))