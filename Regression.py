import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
from datetime import datetime
import statsmodels.api as sm
from scipy import stats
from sklearn import linear_model
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import sklearn.cross_validation as cv
from scipy import stats
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import math


#We have two input files here, one with individual features and the other with combined features after creating dummy variables.

input1=pd.read_csv("D:/Machine Learning/NYCTraffic/inputfile_aggre.csv")
input2=pd.read_csv("D:/Machine Learning/NYCTraffic/inputfile_aggre1.csv")

#Feature extraction
#VarianceThreshold removes the first column, which has a probability p = 0.6 of containing the same value. 
sel = VarianceThreshold(threshold=(.6 * (1 - .6)))
sel.fit_transform(input1)
#For our input, none of them are removed. So all the features are important.

#Feature extraction using chi square test.
#According to this test, the order of importance of features is Zone,Weekday,Hour.
target=input1[['Count']]
data=input1[[col for col in input1.columns if col not in ['Count']]]
X, y = data, target
X_new = SelectKBest(chi2, k=2).fit_transform(X, y)


###As a scaling factor and for better results, we take the target variable 'Count' as log values.

input1['Count'] = np.log10(input1['Count']+1)

#Split data into training and testing files for cross validation.
x_train, x_test, y_train, y_test = cv.train_test_split(data, target, test_size=2.0/10, random_state=0)

###################################Linear Regression with normal features.##################################
#This data is not linearly seperable, so linear regression is not a good fit here. We got low R squared value. 
ols = linear_model.LinearRegression()
ols.fit(x_train, y_train)
'training R^2: %.2f',ols.score(x_train, y_train)
'testing R^2: %.2f',ols.score(x_test, y_test)

'training RMSE:', mean_squared_error(y_train, ols.predict(x_train))
'testing RMSE:', mean_squared_error(y_test, ols.predict(x_test))

cross_val_score(ols, X, y, cv=5)
cross_val_score(linear_model.LinearRegression(alpha=float(alpha), random_state=2), data, target, 'mean_squared_error', cv=5).mean()




mlp = MLPRegressor(solver='lbfgs', hidden_layer_sizes=50,
                           max_iter=300, shuffle=True, random_state=1,
                           activation='relu')
mlp.fit(X, y)

##############RandomForest Regression on normal features##############################
RFR=RandomForestRegressor(max_features=3,n_estimators=300)
RFR.fit(x_train, y_train)
training_accuracy = RFR.score(x_train, y_train)
test_accuracy = RFR.score(x_test, y_test)
np.round(np.power(10,np.column_stack((RFR.predict(x_test),y_test))) - 1,decimals=0).astype(int)
rmse = np.sqrt(mean_squared_error(RFR.predict(x_test),y_test))
np.power(10,rmse)


RFR1=RandomForestRegressor(max_features=14,n_estimators=500)
RFR1.fit(x_train, y_train)
from sklearn.metrics import mean_squared_error
mean_squared_error(y_train, RFR1.predict(x_train))
mean_squared_error(y_test, RFR1.predict(x_test))

##############################################################################


#######Using LinearRegression on the combined features dataset.
#Linear Regression did not fit well even with combined features dataset. 
input2['Count'] = np.log10(input2['Count']+1)
target1=input2[['Count']]
data1=input2[[col for col in input2.columns if col not in ['Count']]]

#Linear Regression with normal features.
x_train1, x_test1, y_train1, y_test1 = cv.train_test_split(data1, target1, test_size=2.0/10, random_state=0)

ols1 = linear_model.LinearRegression()
ols1.fit(x_train1, y_train1)
'training R^2: %.2f',ols1.score(x_train1, y_train1)
'testing R^2: %.2f',ols1.score(x_test1, y_test1)

'training RMSE:', mean_squared_error(y_train1, ols1.predict(x_train1))
'testing RMSE:', mean_squared_error(y_test1, ols1.predict(x_test1))

##############RandomForest Regression on normal features##############################
#Combined features dataset gave a little lower RMSE compared to normal features.
RFR1=RandomForestRegressor(max_features=82,n_estimators=300)
RFR.fit(x_train1, y_train1)
training_accuracy = RFR.score(x_train1, y_train1)
test_accuracy = RFR.score(x_test1, y_test1)
np.round(np.power(10,np.column_stack((RFR.predict(x_test1),y_test1))) - 1,decimals=0).astype(int)
rmse = np.sqrt(mean_squared_error(RFR.predict(x_test1),y_test1))
np.power(10,rmse)
dict_feat_imp = dict(zip(list(x_train1.columns.values),RFR1.feature_importances_))

sorted_features = sorted(dict_feat_imp.items(), key=operator.itemgetter(1), reverse=True)
sorted_features

###############
#Linear Regression did not give accurate results even with normal features and combined features.
#Random Forest Regression gave better results with combined features.


  
