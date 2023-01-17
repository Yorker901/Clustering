# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 09:11:00 2022

@author: Mohd Ariz Khan
"""
# Import the data
import pandas as pd
df = pd.read_csv("EastWestAirlines.csv")
df

# Get information of the dataset
df.info()
df.isnull().any()
print('The shape of our data is:', df.shape)
print(df.describe())
df.head()

# Drop the variable
df_1 = df.drop(['ID'],axis=1)
df_1

# Normalize heterogenous numerical data 
from sklearn.preprocessing import normalize
df_1_norm = pd.DataFrame(normalize(df_1),columns = df_1.columns)
df_1_norm

# Create Dendrograms
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 7))  
dendograms = sch.dendrogram(sch.linkage(df_1_norm,'complete'))

# Create Clusters 
from sklearn.cluster import AgglomerativeClustering
clusters = AgglomerativeClustering(n_clusters = 5, affinity='euclidean', linkage='ward')
clusters


Y = pd.DataFrame(clusters.fit_predict(df_1_norm), columns = ['clustersid'])
Y['clustersid'].value_counts()

# Adding clusters to dataset
df_1['clustersid'] = clusters.labels_
df_1

df_1.groupby('clustersid').agg(['mean']).reset_index()

# Plot Clusters
plt.figure(figsize=(10, 7))  
plt.scatter(df_1['clustersid'],df_1['Balance'], c = clusters.labels_) 

#=============================================================================
# Kmeans Clustering
from sklearn.cluster import KMeans

# Elbow curv
wcss=[]
for i in range(1,11):
    KM = KMeans(n_clusters=i)
    KM.fit(df_1_norm)
    wcss.append(KM.inertia_)

import matplotlib.pyplot as plt
plt.plot(range(1,11),wcss)
plt.title('Elbow curv')
plt.xlabel('number of clusters')
plt.ylabel('WCSS')
plt.show()

# selecting 4 clusters from above scree plot
KM = KMeans(n_clusters = 5)
KM.fit(df_1_norm)
KM.labels_

x = pd.Series(KM.labels_)
df['Clust'] = x
df

df.iloc[:,1:11].groupby(df.Clust).mean()

#============================================================================
# DBSCAN Clustering
from sklearn.cluster import DBSCAN

df = df.iloc[:,1:11]
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

pd.concat([df_1,c1],axis=1)






















