#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv('filename',delimiter = ',', encoding = "latin1")


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


print(data.isnull().sum())


# In[6]:


data = data.dropna()


# In[7]:


print(data.isnull().sum())


# In[9]:


km_cao = KModes(n_clusters=10, init = "Cao", n_init = 3, verbose=1)
fitClusters_cao = km_cao.fit_predict(data)


# In[10]:


fitClusters_cao


# In[11]:


clusterCentroidsDf = pd.DataFrame(km_cao.cluster_centroids_)
clusterCentroidsDf.columns = data.columns


# In[12]:


clusterCentroidsDf


# In[13]:


km_huang = KModes(n_clusters=10, init = "Huang", n_init = 3, verbose=1)
fitClusters_huang = km_huang.fit_predict(data)


# In[14]:


fitClusters_huang


# In[15]:


cost = []
for num_clusters in list(range(1,11)):
    kmode = KModes(n_clusters=num_clusters, init = "Cao", n_init = 3, verbose=1)
    kmode.fit_predict(data)
    cost.append(kmode.cost_)


# In[16]:


y = np.array([i for i in range(1,11,1)])
plt.plot(y,cost)


# In[17]:


km_cao = KModes(n_clusters=10, init = "Cao", n_init = 3, verbose=1)
fitClusters_cao = km_cao.fit_predict(data)


# In[18]:


fitClusters_cao


# In[19]:


data2 = data.reset_index()


# In[20]:


clustersDf = pd.DataFrame(fitClusters_cao)
clustersDf.columns = ['cluster_predicted']
combinedDf = pd.concat([data2, clustersDf], axis = 1).reset_index()
combinedDf = combinedDf.drop(['index', 'level_0'], axis = 1)


# In[21]:


combinedDf.head()


# In[22]:


cluster_0 = combinedDf[combinedDf['cluster_predicted'] == 0]
cluster_1 = combinedDf[combinedDf['cluster_predicted'] == 1]
cluster_3 = combinedDf[combinedDf['cluster_predicted'] == 3]
cluster_4 = combinedDf[combinedDf['cluster_predicted'] == 4]
cluster_5 = combinedDf[combinedDf['cluster_predicted'] == 5]
cluster_6 = combinedDf[combinedDf['cluster_predicted'] == 6]
cluster_7 = combinedDf[combinedDf['cluster_predicted'] == 7]
cluster_8 = combinedDf[combinedDf['cluster_predicted'] == 8]
cluster_9 = combinedDf[combinedDf['cluster_predicted'] == 9]


# In[23]:


cluster_0.to_csv('cluster_ER_0.csv', index=False) 
cluster_3.to_csv('cluster_ER_3.csv', index=False) 
cluster_5.to_csv('cluster_ER_4.csv', index=False) 


# In[35]:


plt.subplots(figsize = (20,15))
sns.countplot(x=combinedDf['Assigned Support Org'],order=combinedDf['Assigned Support Org'].value_counts().index,hue=combinedDf['cluster_predicted'])
plt.show()


# In[ ]:


plt.subplots(figsize = (15,5))
sns.countplot(x=combinedDf['KB_number'],order=combinedDf['KB_number'].value_counts().index,hue=combinedDf['cluster_predicted'])
plt.show()


# In[ ]:


plt.subplots(figsize = (15,5))
sns.countplot(x=combinedDf['Incident_ID'],order=combinedDf['Incident_ID'].value_counts().index,hue=combinedDf['cluster_predicted'])
plt.show()


# In[ ]:


corr_matrix = combinedDf.corr()


# In[ ]:


corr_matrix["No_of_Reassignments"].sort_values(ascending=False)


# In[ ]:




