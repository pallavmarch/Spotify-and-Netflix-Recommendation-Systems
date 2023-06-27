#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as np


# In[22]:


data=pd.read_csv("C:/Users/palla/Documents/Py files/datasets/top50.csv")
data.head()


# In[23]:


#checking NA cells
data.isna().sum()


# In[24]:


data['Genre'].value_counts()
#dance pop genre has the most songs in the list


# In[25]:


len(data['Genre'].unique())
#Total genre present


# New Table

# In[26]:


data.columns


#  

# The Popularity graphs

# In[27]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[28]:


plt.figure(figsize=(200, 50))
ax = sns.barplot(x=data['Genre'], y=data['Popularity'])

# Increase font size of values -- Google search
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 20), textcoords = 'offset points', fontsize=100)

plt.title('Genre vs Popularity', fontsize=200)
plt.xlabel('Genre', fontsize=150)
plt.ylabel('Popularity',fontsize=150)
plt.tick_params(axis='x', labelsize=100, rotation=75)  
plt.show()


# In[29]:


plt.figure(figsize=(200, 50))
ax = sns.barplot(x=data['Artist.Name'], y=data['Popularity'])

# Increase font size of values -- Google search
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 20), textcoords = 'offset points', fontsize=100)

plt.title('Artist Name vs Popularity', fontsize=200)
plt.xlabel('Artist Name', fontsize=150)
plt.ylabel('Popularity',fontsize=150)
plt.tick_params(axis='x', labelsize=100, rotation=75)  
plt.show()


#  

# Heat Map

# In[30]:


#Creating a new column for Loudness since the values were in negative
data['Loudness+']=-(data['Loudness..dB..'])
data['Loudness+'].unique()


# In[31]:


plt.subplots(figsize=(8, 7))
da1=data[['Beats.Per.Minute', 'Energy',
       'Danceability', 'Loudness+', 'Liveness', 'Valence.', 'Length.',
       'Acousticness..', 'Speechiness.', 'Popularity']]
sns.heatmap(da1.corr(), annot=True)
plt.show()


# Beats Per Minute greatly correlates with Speechiness

#  

#  

# Using KMeans clusterization with 10 clusters 

# In[32]:


from sklearn.cluster import KMeans

num_types = [ 'int64', 'float64'] 
#This list cointains the data types which will be considered for clustering.
xy = data.select_dtypes(include=num_types) 
#Selects the columns from the dataset with the data types which are matching with the 'num_types' and places them in xy


km = KMeans(n_clusters=10)  #We'll have 10 distinct groups from the algorithm
k=km.fit_predict(xy)  #a cluster label is assigned to each data point
data['no']=k


# In[33]:


data['no'].unique()


#  

# Recommendation System

# In[34]:


from tqdm import tqdm


# In[35]:


data['name']=data['Track.Name']


# In[36]:


class SpotifyRecomm():
    def __init__(self, rec_data):
        self.rec_data_ = rec_data
    
    def change_data(self, rec_data):
        self.rec_data_ = rec_data
    
    def get_recomm(self, song_name, amount=1):
        distances = []
        #choosing the data for our song
        song = self.rec_data_[(self.rec_data_.name.str.lower() == song_name.lower())].head(1).values[0]
        #dropping the data with our song
        res_data = self.rec_data_[self.rec_data_.name.str.lower() != song_name.lower()]
        for r_song in tqdm(res_data.values):
            dist = 0
            for col in np.arange(len(res_data.columns)):
                #indeces of non-numerical columns
                if not col in [0, 1, 2, 3, 16]:
                    #calculating the manhettan distances for each numerical feature
                    dist = dist + np.absolute(float(song[col]) - float(r_song[col]))
            distances.append(dist)
        res_data['distance'] = distances
        #sorting our data to be ascending by 'distance' feature
        res_data = res_data.sort_values('distance')
        columns = ['Artist.Name', 'Track.Name']
        return res_data[columns][:amount]
    
    
recommender = SpotifyRecomm(data)

recommender.get_recomm('China', 5)


# In[37]:


recommender.get_recomm('Señorita', 5)

