#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


#Read business object and display it's columns
business_obj_loc = 'yelp_dataset/yelp_academic_dataset_business.json'
businessd_df = pd.read_json(business_obj_loc, lines=True)
columns = businessd_df.head()
columns 


# In[6]:


#Find missing values in business object
businessd_df.isnull().sum()


# In[7]:


#Show number of open and closed businesses
df_isopen = businessd_df[['is_open','business_id']].groupby(['is_open']).count()
df_isopen


# In[8]:


#Open vs closed businesses
sns.countplot(x='is_open',data=businessd_df)


# In[9]:


#Frequency of ratings
sns.countplot(x='stars',data=businessd_df)


# In[17]:


#Nuber of businesses by city
df_by_city = businessd_df[['city','business_id']].groupby(['city']).count()
df_by_city.sort_values(by=['business_id'],ascending=False, inplace=True)
df_by_city[df_by_city['business_id']>1000]


# In[35]:


#Show number businesses by category
df_categories = businessd_df[['categories','business_id']].groupby(['categories']).count()
df_categories.sort_values(by=['business_id'],ascending=False, inplace=True)
df_categories[df_categories['business_id']>100]


# In[54]:


#Find number of restaurants
restaurant_df = businessd_df[businessd_df['categories'].str.contains('Restaurant',case=False, na=False)]
restaurant_df = restaurant_df[['categories','business_id']].groupby(['categories']).count()
restaurant_df.sort_values(by=['business_id'],ascending=False, inplace=True)
restaurant_df.count()

