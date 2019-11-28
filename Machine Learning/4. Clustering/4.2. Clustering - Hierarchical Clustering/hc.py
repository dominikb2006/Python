# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 10:57:10 2019

@author: BrysDom
"""
#Hierarchical Clustering

#%reset -f
#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

#Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X,method='ward'))

plt.title('dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

#Fitting hierarchical clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc = hc.fit_predict(X)

#Visualising the clusters
plt.scatter(X[y_hc==0,0],X[y_hc==0,1], s=100, c='red',label='Careful')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1], s=100, c='blue',label='Standard')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1], s=100, c='green',label='Targets')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1], s=100, c='cyan',label='Careless')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1], s=100, c='magenta',label='Sensible')
plt.scatter(X[y_hc==5,0],X[y_hc==5,1], s=100, c='black',label='Careless')
plt.scatter(X[y_hc==6,0],X[y_hc==6,1], s=100, c='pink',label='Sensible')
plt.scatter(y_hc.cluster_centers_[:,0],y_hc.cluster_centers_[:,1], s=300, c='yellow',label='Centroids')
plt.title('Clusters of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

