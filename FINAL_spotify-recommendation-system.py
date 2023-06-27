#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data=pd.read_csv("C:/Users/palla/Documents/Py files/datasets/Spotify_dataset.csv")
data.head()


# In[3]:


#checking NA cells
data.isna().sum()


# In[4]:


data.shape


# In[5]:


data.columns


# In[6]:


data['Genre'].value_counts()


# In[7]:


data['Artist.Name'].value_counts()
#The Beatles with the highest


# In[8]:


data['Genre'].unique()


# In[9]:


len(data['Genre'].unique())
#Total genre present


# In[10]:


len(data['Artist.Name'].unique())


#  

# Updating the dataset

# In[11]:


data.columns


#  

# In[12]:


data=data[['Track.Name', 'Artist.Name', 'album_name','Genre', 'Popularity', 'Length.', 'explicit', 'Danceability', 'Energy', 'key',
       'Loudness..dB..', 'mode', 'Speechiness.', 'Acousticness..', 'Liveness',
       'Valence.', 'tempo', 'time_signature']]


# In[13]:


data['Loudness..dB..'].unique()


# In[14]:


#Since Loudness..dB.. values are in negative, we are changing them to positive
data['Loudness..dB..']=-(data['Loudness..dB..'])
data['Loudness..dB..'].unique()


# In[15]:


data['Loudness..dB..'].max()


# In[16]:


data['Loudness..dB..'].min()


# In[17]:


data.head()


#   

# The Popularity graphs

# In[18]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[19]:


plt.figure(figsize=(300, 70))
ax = sns.barplot(x=data['Genre'], y=data['Popularity'])

# Increase font size of values -- Google search
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 20), textcoords = 'offset points', fontsize=100)

plt.title('Genre vs Popularity', fontsize=200)
plt.xlabel('Genre', fontsize=150)
plt.ylabel('Popularity',fontsize=150)
plt.tick_params(axis='x', labelsize=100, rotation=90)  
plt.show()


# Pop-Film has the highest populatity, followed by K-Pop

# In[20]:


#Creating new dataset for Artists
aa=data[['Artist.Name','Track.Name','Popularity']]
aa=data.groupby(by=['Artist.Name']).mean()[['Popularity']]
aa = aa[aa['Popularity'] > 87]
aa.sort_values('Popularity',ascending=True)


# In[21]:


plt.figure(figsize=(300, 70))
ay=sns.barplot(x=aa.index, y='Popularity', data=aa)

# Increase font size of values -- Google search
for p in ay.patches:
    ay.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 2), textcoords = 'offset points', fontsize=100)

plt.title('Average Popularity of Artists with Popularity > 87',fontsize=200)
plt.xlabel('Artist',fontsize=100)
plt.ylabel('Popularity',fontsize=100)
plt.tick_params(axis='x', labelsize=100, rotation=90)  
plt.show()


#  

# Heat Map

# In[22]:


data.columns


# In[23]:


plt.subplots(figsize=(14, 12))
da1=data[['Popularity',
       'Length.', 'explicit', 'Danceability', 'Energy', 'key',
       'Loudness..dB..', 'mode', 'Speechiness.', 'Acousticness..', 'Liveness',
       'Valence.', 'tempo', 'time_signature']]
sns.heatmap(da1.corr(), annot=True)
plt.show()


# Loudness greatly correlates with Acousticness

#  

#  

# Using KMeans clusterization with 10 clusters 

# In[24]:


from sklearn.cluster import KMeans

num_types = [ 'int64', 'float64'] 
#This list cointains the data types which will be considered for clustering.
xy = data.select_dtypes(include=num_types) 
#Selects the columns from the dataset with the data types which are matching with the 'num_types' and places them in xy


km = KMeans(n_clusters=10)  #We'll have 10 distinct groups from the algorithm
k=km.fit_predict(xy)  #a cluster label is assigned to each data point
data['no']=k


# In[25]:


data['no'].unique()


#  

# Recommendation System

# In[26]:


from tqdm import tqdm


# In[27]:


#Rename column
data = data.rename(columns={'Track.Name': 'Trackname'})
data.columns


# In[28]:


class SptifyRec():
    def __init__(self, rec_data):
        self.rec_data_ = rec_data
    
    def change_data(self, rec_data):
        self.rec_data_ = rec_data
    
    def get_recomm(self, song_name, amount=1):
        distances = []
        #choosing the data for our song
        song = self.rec_data_[(self.rec_data_.Trackname.str.lower() == song_name.lower())].head(1).values[0]
        #dropping the data with our song
        res_data = self.rec_data_[self.rec_data_.Trackname.str.lower() != song_name.lower()]
        for r_song in tqdm(res_data.values):
            dist = 0
            for col in np.arange(len(res_data.columns)):
                #indeces of non-numerical columns
                if not col in [0, 1, 2, 3,4, 19]:
                    #calculating the manhettan distances for each numerical feature
                    dist = dist + np.absolute(float(song[col]) - float(r_song[col]))
            distances.append(dist)
        res_data['distance'] = distances
        #sorting our data to be ascending by 'distance' feature
        res_data = res_data.sort_values('distance')
        columns = ['Artist.Name', 'Trackname', 'album_name', 'Genre']
        return res_data[columns][:amount]
    
 
#Object
recommender = SptifyRec(data)
recommender.get_recomm('Ghost - Acoustic', 5)


# In[30]:


recommender.get_recomm('Water Into Light', 5)


# In[40]:


print("\N{smiling face with sunglasses}"*35)

