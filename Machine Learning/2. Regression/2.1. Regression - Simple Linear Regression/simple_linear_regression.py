# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 15:36:37 2019

@author: BrysDom
"""
#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#Splitting database into Training and Test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 1/3,random_state = 0)

##Feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#Fitting Linear Regression to Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predicting Test result
y_pred = regressor.predict(X_test)

#Visualising Training results
plt.scatter(X_train,y_train,color='red')
plt.scatter(X_test,y_test,color='green')
plt.plot(X_train,regressor.predict(X_train), color='blue')
#plt.scatter(X_test,y_test,color='yellow')
#plt.plot(X_test,regressor.predict(X_test), color='green')
plt.title('Salary vs Experience (Training(red)/Test(green) set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()