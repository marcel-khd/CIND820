#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


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


# In[3]:


#Read a selected city reviews from csv file 
#Prerequisite: run Recommender.ipynb
reviews_selected_Restaurants = pd.read_csv('reviews_selected_city_restaurants_Clean.csv')


# In[4]:


reviews_selected_Restaurants.head()


# In[5]:


#Nuber of reviews by business
reviews_by_business = reviews_selected_Restaurants[['business_id','review_id']].groupby(['business_id']).count()
reviews_by_business.sort_values(by=['review_id'],ascending=False, inplace=True)
reviews_by_business = reviews_by_business.rename(columns={'review_id': 'nbr_of_reviews'})
reviews_by_business[reviews_by_business['nbr_of_reviews']>2000]


# In[6]:


#Frequency of reviews
#reviews_by_business['range'] = pd.cut(reviews_by_business.nbr_of_reviews, bins=[0,11,101,1001,10001], labels=["0-10","11-100","101-1000","1001+"])
reviews_by_business['range'] = pd.cut(reviews_by_business.nbr_of_reviews, bins=[0,51,101,151,10000], labels=["0-50","51-100","101-150","151+"])
#reviews_by_business['range'] = pd.cut(reviews_by_business.nbr_of_reviews,50)
sns.countplot(x='range',data=reviews_by_business)


# In[7]:


#Frequency of reviews
reviews_by_business_grouped = reviews_by_business[['range','nbr_of_reviews']].groupby(['range']).count()
reviews_by_business_grouped = reviews_by_business_grouped.rename(columns={'nbr_of_reviews': 'nbr_of_business'})
#reviews_by_business_grouped.sort_values(by=['nbr_of_reviews'],ascending=False, inplace=True)
reviews_by_business_grouped


# In[39]:


#Desciptive statistics about number of reviews by business
reviews_by_business['nbr_of_reviews'].describe()


# In[8]:


#Note: Number of businesses decreases exponentially with number of reviews
hist = reviews_by_business[reviews_by_business['nbr_of_reviews']<600].hist(bins=61)


# In[9]:


hist = reviews_by_business[reviews_by_business['nbr_of_reviews'] <=50].hist()


# In[10]:


#Nuber of reviews by year
reviews_selected_Restaurants['year'] = reviews_selected_Restaurants['date'].str[:4]
reviews_by_year = reviews_selected_Restaurants[['year','review_id']].groupby(['year']).count()
reviews_by_year = reviews_by_year.rename(columns={'review_id': 'nbr_of_reviews'})
reviews_by_year


# In[11]:


#Number of reviews by year: 2018-2019 have highest number of reviews
#There is a drop in 2020 due to Covid
reviews_by_year = reviews_by_year.reset_index()
sns.barplot(data=reviews_by_year,x='year',y='nbr_of_reviews')


# In[27]:


#Find number of characters in a review
def count_text(string_input):
    string_input = str(string_input)
    nbr_chars = len(string_input)
    return nbr_chars


reviews_selected_Restaurants['nbr_characters'] = reviews_selected_Restaurants['text'].apply(count_text)
hist = reviews_selected_Restaurants['nbr_characters'].hist()


# In[38]:


#Desciptive statistics about number of characters
#Note that minimum number of charactes is 3, with mean = 360
#Number of characters in 1st quartile = 149
reviews_selected_Restaurants['nbr_characters'].describe()
