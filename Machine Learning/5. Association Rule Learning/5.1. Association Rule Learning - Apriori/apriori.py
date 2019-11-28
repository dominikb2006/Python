# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 14:31:00 2019

@author: BrysDom
"""
#Apriori
#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv',header=None)
transactions =[]
for i in range (0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])

"""for i in range (0,7501):
    for j in range(0,20):
        transactions.append([str(dataset.values[i,j])])"""
     
#Training Aprior on the dataset
from apyori import apriori
rules = apriori(transactions,min_support=0.003,min_confidence=0.2,min_lift=3, min_length=2)

#Visualising results
results = list(rules)
#results[0][0]