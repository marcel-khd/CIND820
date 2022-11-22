#!/usr/bin/env python
# coding: utf-8

# # 1. Load Data

# In[1]:


import numpy as np
import pandas as pd


# In[3]:


#Read business object
business_obj_loc = 'yelp_dataset/yelp_academic_dataset_business.json'
business_df = pd.read_json(business_obj_loc, lines=True)
columns = business_df.head()
columns 


# In[4]:


#Replace null values in business df
business_df.fillna('NA', inplace=True)


# In[5]:


#Select businesses in Nashville only
df_City = business_df[business_df['city']=='Nashville']
#Show number of open and closed businesses
df_City[['is_open','business_id']].groupby(['is_open']).count()


# In[6]:


#Select open businesses only
df_City = df_City[df_City['is_open']==1]
count_row = df_City.shape[0]
print(str(count_row))


# In[7]:


#Find categories that contain the keyword 'Restaurants'
df_City_Restaurants = df_City[df_City['categories'].str.contains('Restaurants')]
count_row = df_City_Restaurants.shape[0]
print(str(count_row))


# In[8]:


#Show number restaurants by category in the selected city
df_categories = df_City_Restaurants[['categories','business_id']].groupby(['categories']).count()
#df_categories.sort_values(by=['business_id'],ascending=False, inplace=True)
#df_categories[df_categories['business_id']>10]


# In[9]:


#Save selected city restaurants categories to csv file
df_categories.to_csv('categories_selected_city_restaurants.csv', sep=',')


# In[9]:


#Read reviews object and display it's columns

b_pandas = []
r_dtypes = {"stars": np.float16, 
            "useful": np.int32, 
            "funny": np.int32,
            "cool": np.int32,
           }

review_obj_loc = 'yelp_dataset/yelp_academic_dataset_review.json'
reader = pd.read_json(review_obj_loc, orient="records", lines=True, dtype=r_dtypes, chunksize=1000)

for chunk in reader:
    reduced_chunk = chunk[chunk['business_id'].isin(df_City_Restaurants['business_id'])]
    reduced_chunk = reduced_chunk.replace(',',';')
    b_pandas.append(reduced_chunk)
    
reviews_City_Restaurants = pd.concat(b_pandas, ignore_index=True)

#Write File to csv
reviews_City_Restaurants.to_csv('reviews_selected_city_restaurants.csv', sep=',')

count_row = reviews_City_Restaurants.shape[0]
print('There are  ' + str(count_row) + ' reviews for restaurants in selected city')


# # 2. Text Pre-Processing

# In[78]:


import string
import re
import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer


# In[16]:


nltk.download('stopwords')


# In[11]:


def clean_text(string_input):
    ## Remove puncuation
    string_input = string_input.translate(string.punctuation)
    
    ## Convert words to lower case and split them
    string_input = string_input.lower().split()
    
    ## Remove stop words
    stops = set(stopwords.words("english"))
    string_input = [w for w in string_input if not w in stops and len(w) >= 3]
    
    string_input = " ".join(string_input)
    
    # Clean the text
    string_input = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", string_input)
    string_input = re.sub(r"what's", "what is ", string_input)
    string_input = re.sub(r"\'s", " ", string_input)
    string_input = re.sub(r"\'ve", " have ", string_input)
    string_input = re.sub(r"n't", " not ", string_input)
    string_input = re.sub(r"i'm", "i am ", string_input)
    string_input = re.sub(r"\'re", " are ", string_input)
    string_input = re.sub(r"\'d", " would ", string_input)
    string_input = re.sub(r"\'ll", " will ", string_input)
    string_input = re.sub(r",", " ", string_input)
    string_input = re.sub(r"\.", " ", string_input)
    string_input = re.sub(r"!", " ! ", string_input)
    string_input = re.sub(r"\/", " ", string_input)
    string_input = re.sub(r"\^", " ^ ", string_input)
    string_input = re.sub(r"\+", " + ", string_input)
    string_input = re.sub(r"\-", " - ", string_input)
    string_input = re.sub(r"\=", " = ", string_input)
    string_input = re.sub(r"'", " ", string_input)
    string_input = re.sub(r":", " : ", string_input)
    string_input = re.sub(r"e - mail", "email", string_input) 
    return string_input


# In[12]:


#Clean up comments
reviews_City_Restaurants['text'] = reviews_City_Restaurants['text'].apply(clean_text)


# In[13]:


#Save Phiiladelphia restaurants reviews to csv file
reviews_City_Restaurants.to_csv('reviews_selected_city_restaurants_Clean.csv', sep=',')


# In[37]:


#Read Phiiladelphia restaurants reviews from csv file
reviews_City_Restaurants = pd.read_csv('reviews_selected_city_restaurants_Clean.csv')


# In[38]:


#Find number of characters in a review
def count_text(string_input):
    string_input = str(string_input)
    nbr_chars = len(string_input)
    return nbr_chars


reviews_City_Restaurants['nbr_characters'] = reviews_City_Restaurants['text'].apply(count_text)


# In[52]:


#Select a subset of reviews for better quality
#Select reviews within the last 5 years
reviews_City_Restaurants['year'] = reviews_City_Restaurants['date'].str[:4]
reviews_City_Restaurants_reduced = reviews_City_Restaurants[reviews_City_Restaurants['year'].isin(['2018','2019','2020','2021','2022'])]

#Select reviews with at least 50 characters
reviews_City_Restaurants_reduced = reviews_City_Restaurants_reduced[reviews_City_Restaurants['nbr_characters']>49]


# In[53]:


reviews_City_Restaurants_reduced.shape


# In[74]:


#Group the reviews by business
reviews_grouped_by_business = reviews_City_Restaurants_reduced.drop(['row_nbr','review_id','user_id','stars','useful','funny','cool','date','year','nbr_characters'], axis=1)
reviews_grouped_by_business.reset_index()

reviews_grouped_by_business['reviews'] = reviews_grouped_by_business.groupby(['business_id'])['text'].transform(lambda x : ' | '.join(x))

# drop duplicate data
reviews_grouped_by_business = reviews_grouped_by_business.drop(['text'], axis=1)
reviews_grouped_by_business = reviews_grouped_by_business.drop_duplicates()

reviews_grouped_by_business.shape


# In[77]:


reviews_grouped_by_business.head()


# In[79]:


#Vectorize reviews
count_vectorizer = CountVectorizer(min_df = .01,max_df = .99, tokenizer = WordPunctTokenizer().tokenize)
reviews_grouped_by_business = reviews_grouped_by_business.fillna('')
vectorized_reviews = count_vectorizer.fit_transform(reviews_grouped_by_business['reviews'])


# In[80]:


vectorized_reviews.shape


# # 3. Randomly select "seed" restaurant

# In[81]:


#Randomly select a restaurant
print(df_City_Restaurants.iloc[200,:])
print(df_City_Restaurants.iloc[200,12])


# In[82]:


#Find all the seed restaurant's reviews by copy/paste business ID to this command
selected_reviews = reviews_City_Restaurants_reduced[reviews_City_Restaurants_reduced['business_id']=='HCHHrf21UAgbxAi8T4Q4Iw']
selected_reviews = selected_reviews['text']
selected_reviews.head()


# In[83]:


#Find how many reviews the seed restaurant has
print('There are  ' + str(selected_reviews.shape[0]) + ' reviews for the selected restaurant')


# # 4. Calculate recommendations

# In[84]:


from scipy.spatial.distance import cdist
# find most similar reviews
distance = cdist(count_vectorizer.transform(selected_reviews).todense().mean(axis=0), 
              vectorized_reviews.todense(),metric='cosine')

distance


# In[89]:


distance.size


# In[85]:


most_similar = distance.argsort().ravel()[:10]
most_similar


# In[95]:


df_most_similar = df_City_Restaurants.loc[df_City_Restaurants['business_id'].isin(reviews_grouped_by_business['business_id'].iloc[most_similar]), ['business_id', 'categories', 'name', 'stars','review_count']]
df_most_similar


# # 5. Initial observations
# Seed restaurant category is: American (New), Tacos, Mexican, Restaurants
# Recommended restaurants are all within the seed restaurant's categories, so they are adequately similar
