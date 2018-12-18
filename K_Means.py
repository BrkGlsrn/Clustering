# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 14:54:27 2018

@author: TCBGULSEREN
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

veriler = pd.read_csv('musteriler.csv')
X = veriler.iloc[:,3:].values

#İlk cluster sayısı 3 ile denendi
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters =3 , init='k-means++')
kmeans.fit(X)
print(kmeans.cluster_centers_)

sonuclar = []
for i in range(1,10):
    kmeans = KMeans(n_clusters = i , init = 'k-means++',random_state=123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)

plt.plot(range(1,10),sonuclar)    
plt.show()

#En iyi hiyerarşi 4 cluster (Grafik Dirsek Noktası)
kmeans = KMeans(n_clusters = 4 , init = 'k-means++',random_state=123)
y_tahmin = kmeans.fit_predict(X)
print(y_tahmin)

plt.scatter(X[y_tahmin==0,0],X[y_tahmin==0,1],s=100,c='red')
plt.scatter(X[y_tahmin==1,0],X[y_tahmin==1,1],s=100,c='blue')
plt.scatter(X[y_tahmin==2,0],X[y_tahmin==2,1],s=100,c='green')
plt.scatter(X[y_tahmin==3,0],X[y_tahmin==3,1],s=100,c='orange')



















