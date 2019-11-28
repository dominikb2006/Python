# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 12:36:43 2019

@author: BrysDom
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#NLP - Natural Language Processing

#Importing dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

#Cleaning text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for _ in range(0,len(dataset)):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][_])
    review = review.lower()
    review = review.split()
    ps=PorterStemmer()
    #review = [word for word in review if not word in set(stopwords.words('english'))]
    review= [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

#Creating Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y =  dataset.iloc[:,1].values

#RES={}

TP=cm[0,0]
TN=cm[1,1]
FP=cm[0,1]
FN=cm[1,0]
Accuracy=(TP+TN)/(TP+TN+TP+FN)
Precision = TP/(TP+FP)
Recall=TP/(TP+FN)
F1_Score = 2*Precision*Recall/(Precision+Recall)
RES[R_name]=F1_Score





#Naive Bayes
"""R_name='Naive Bayes'
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)
 
#Predicting the Test set results
y_pred = classifier.predict(X_test)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)"""


#Decision Tree Classification
"""R_name='Decision Tree Classification'
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Fitting Decision Tree to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)
 
#Predicting the Test set results
y_pred = classifier.predict(X_test)

#Mkaing the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)"""


#Random Forest Classification
"""R_name='Random Forest Classification'
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Fitting Random Forest to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)
 
#Predicting the Test set results
y_pred = classifier.predict(X_test)

#Mkaing the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)"""


#Kernel SVM
"""R_name='Kernel SVM'
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel='rbf',random_state=0)
classifier.fit(X_train,y_train)
 
#Predicting the Test set results
y_pred = classifier.predict(X_test)

#Mkaing the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)"""

#SVM
"""R_name='SVM'
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Fitting SVM to the Training set 
from sklearn.svm import SVC
classifier = SVC(kernel='rbf',random_state=0)
classifier.fit(X_train,y_train)
 
#Predicting the Test set results
y_pred = classifier.predict(X_test)

#Mkaing the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)"""


#KNN
"""R_name='KNN'
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Fitting Logistic Regression to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(X_train,y_train)
 
#Predicting the Test set results
y_pred = classifier.predict(X_test)

#Mkaing the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)"""


#Logistic regression
"""R_name='Logistic regression'
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)
 
#Predicting the Test set results
y_pred = classifier.predict(X_test)

#Mkaing the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)"""
