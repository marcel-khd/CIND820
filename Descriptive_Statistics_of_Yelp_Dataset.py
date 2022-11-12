#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Read business object and display it's columns
business_obj_loc = 'yelp_dataset/yelp_academic_dataset_business.json'
business_df = pd.read_json(business_obj_loc, lines=True)
columns = business_df.head()
columns 


# In[4]:


#Find missing values in business object
business_df.isnull().sum()


# In[5]:


#Show number of open and closed businesses
df_isopen = business_df[['is_open','business_id']].groupby(['is_open']).count()
df_isopen


# In[6]:


#Open vs closed businesses
sns.countplot(x='is_open',data=business_df)


# In[7]:


#Frequency of ratings
sns.countplot(x='stars',data=business_df)


# In[7]:


#Nuber of businesses by city
df_by_city = business_df[['city','business_id']].groupby(['city']).count()
df_by_city.sort_values(by=['business_id'],ascending=False, inplace=True)
df_by_city[df_by_city['business_id']>1000]


# In[8]:


#Nuber of businesses by state
df_by_state = business_df[['state','business_id']].groupby(['state']).count()
df_by_state.sort_values(by=['business_id'],ascending=False, inplace=True)
df_by_state


# In[9]:


#Show number businesses by category
df_categories = business_df[['categories','business_id']].groupby(['categories']).count()
df_categories.sort_values(by=['business_id'],ascending=False, inplace=True)
df_categories[df_categories['business_id']>400]


# In[10]:


#Find number of restaurants
restaurant_df = business_df[business_df['categories'].str.contains('Restaurant',case=False, na=False)]
restaurant_df = restaurant_df[['categories','business_id']].groupby(['categories']).count()
restaurant_df.sort_values(by=['business_id'],ascending=False, inplace=True)
restaurant_df.count()


# In[ ]:


#Read reviews object and display it's columns
#This method reads the whole json object at once, but it doesn't work due to memory issue
#r_dtypes = {"stars": np.float16, 
#            "useful": np.int32, 
#            "funny": np.int32,
#            "cool": np.int32,
#           }
     
#review_obj_loc = 'yelp_dataset/yelp_academic_dataset_review.json'
#reviews_df = pd.read_json(review_obj_loc, orient="records", lines=True, dtype=r_dtypes)
#columns = review_df.head()
#columns 


# In[2]:


#Read reviews object and display it's columns
#First check number of reviews pre-2016
b_pandas = []
r_dtypes = {"stars": np.float16, 
            "useful": np.int32, 
            "funny": np.int32,
            "cool": np.int32,
           }

review_obj_loc = 'yelp_dataset/yelp_academic_dataset_review.json'
reader = pd.read_json(review_obj_loc, orient="records", lines=True, dtype=r_dtypes, chunksize=1000)

for chunk in reader:
    reduced_chunk = chunk.query("`date` < '2016-01-01'")
    reduced_chunk = reduced_chunk.replace(',',';')
    b_pandas.append(reduced_chunk)
    
reviews_df_pre_2016 = pd.concat(b_pandas, ignore_index=True)

#Write File to csv
reviews_df_pre_2016.to_csv('reviews_df_pre_2016', sep=',')

count_row = reviews_df_pre_2016.shape[0]
print('There are  ' + str(count_row) + ' reviews before 2016-01-01')


# In[3]:


#Read reviews object and display it's columns
#Check number of reviews between-2016 and 2018
b_pandas = []
r_dtypes = {"stars": np.float16, 
            "useful": np.int32, 
            "funny": np.int32,
            "cool": np.int32,
           }

review_obj_loc = 'yelp_dataset/yelp_academic_dataset_review.json'
reader = pd.read_json(review_obj_loc, orient="records", lines=True, dtype=r_dtypes, chunksize=1000)

for chunk in reader:
    reduced_chunk = chunk.query("`date` >= '2016-01-01' and `date` < '2018-01-01'")
    reduced_chunk = reduced_chunk.replace(',',';')
    b_pandas.append(reduced_chunk)
    
reviews_df_2016_2018 = pd.concat(b_pandas, ignore_index=True)

#Write File to csv
reviews_df_2016_2018.to_csv('reviews_2016_2018', sep=',')

count_row = reviews_df_2016_2018.shape[0]
print('There are  ' + str(count_row) + ' reviews after 2016-01-01 and before 2018-01-01')


# In[2]:


#Read reviews object and display it's columns
#Check number of reviews after 2018
b_pandas = []
r_dtypes = {"stars": np.float16, 
            "useful": np.int32, 
            "funny": np.int32,
            "cool": np.int32,
           }

review_obj_loc = 'yelp_dataset/yelp_academic_dataset_review.json'
reader = pd.read_json(review_obj_loc, orient="records", lines=True, dtype=r_dtypes, chunksize=1000)

for chunk in reader:
    reduced_chunk = chunk.query("`date` >= '2018-01-01'")
    reduced_chunk = reduced_chunk.replace(',',';')
    b_pandas.append(reduced_chunk)
    
reviews_df_2018 = pd.concat(b_pandas, ignore_index=True)

#Write File to csv
reviews_df_2018.to_csv('reviews_df_2018', sep=',')

count_row = reviews_df_2018.shape[0]
print('There are  ' + str(count_row) + ' reviews on or after 2018-01-01')


# In[3]:


columns = reviews_df_2018.head()
columns 
