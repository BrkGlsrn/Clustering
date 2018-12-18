# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 15:49:15 2018

@author: TCBGULSEREN
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

veriler = pd.read_csv('musteriler.csv')

X = veriler.iloc[:,3:].values

from sklearn.cluster import AgglomerativeClustering
ac =AgglomerativeClustering(n_clusters =3 , affinity ='euclidean',linkage ='ward')
y_tahmin = ac.fit_predict(X)
print(y_tahmin)

plt.scatter(X[y_tahmin==0,0],X[y_tahmin==0,1],s=100,c='red')
plt.scatter(X[y_tahmin==1,0],X[y_tahmin==1,1],s=100,c='blue')
plt.scatter(X[y_tahmin==2,0],X[y_tahmin==2,1],s=100,c='green')
plt.show()

#Farklı kütüphaneden import dendogram için
import scipy.cluster.hierarchy as sch
dendogram =sch.dendrogram(sch.linkage(X , method ='ward'))
plt.show()