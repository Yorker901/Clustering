# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 01:07:16 2022

@author: Mohd Ariz Khan
"""
# import the data
import pandas as pd
df = pd.read_csv("crime_data.csv")
df

# Normalized data fuction
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)

df_norm = norm_func(df.iloc[:,1:])
df_norm


# Dendogram
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(df_norm,method='average'))

# create clusters
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'complete')
cluster

Y = cluster.fit_predict(df_norm)
Y

df['h_clusterid'] = cluster.labels_
df
#=====================================================================================

# Kmeans Clustering
from sklearn.cluster import KMeans

# Elbow curv
wcss=[]
for i in range(1,11):
    KM = KMeans(n_clusters=i)
    KM.fit(df_norm)
    wcss.append(KM.inertia_)

import matplotlib.pyplot as plt
plt.plot(range(1,11),wcss)
plt.title('Elbow curv')
plt.xlabel('number of clusters')
plt.ylabel('WCSS')
plt.show()

# selecting 4 clusters from above scree plot
KM = KMeans(n_clusters = 4)
KM.fit(df_norm)
KM.labels_

x = pd.Series(KM.labels_)
df['Clust'] = x
df

df.iloc[:,1:5].groupby(df.Clust).mean()
#============================================================================

# DBSCAN Clustering
from sklearn.cluster import DBSCAN

df = df.iloc[:,1:5]
df.values

# Normalize heterogenous numerical data using standard scalar fit transform to dataset
from sklearn.preprocessing import StandardScaler
SS = StandardScaler().fit(df.values)
x = SS.transform(df.values)

DBS = DBSCAN(eps=2,min_samples=4)
DBS.fit(x)

# Noisy samples are given the label -1.
DBS.labels_

# Adding clusters to dataset
c1 = pd.DataFrame(DBS.labels_,columns=['cluster'])
c1

pd.concat([df,cl],axis=1)






















