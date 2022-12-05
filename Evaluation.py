#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import random
from nltk.tokenize import WordPunctTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import cdist
import itertools
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#Define a function to calculate similarity between two strings
import difflib
def string_similarity(str1, str2):
    result =  difflib.SequenceMatcher(a=str1.lower(), b=str2.lower())
    return result.ratio()


# # 1. Load Pre-Processed Data

# In[3]:


#Read selected city restaurants from csv file - see Recommender notebook
df_City_Restaurants = pd.read_csv('selected_city_restaurants.csv')
#Reset index to make sure we are getting the correct row number for the seed restaurant
df_City_Restaurants = df_City_Restaurants.loc[:,['business_id', 'categories', 'name', 'stars','review_count']]
df_City_Restaurants.reset_index(drop=True)


# In[4]:


#Read pre-processed restaurants reviews from csv file - see Recommender notebook
reviews_City_Restaurants_reduced = pd.read_csv('reviews_selected_city_restaurants_Clean_Reduced.csv')
#Reset index to make sure we are getting the correct row number for the seed restaurant
reviews_City_Restaurants_reduced = reviews_City_Restaurants_reduced.loc[:,['user_id','business_id', 'text']]
reviews_City_Restaurants_reduced.reset_index(drop=True)


# # 2. Vectorize reviews

# In[5]:


#Group the reviews by business
reviews_grouped_by_business = reviews_City_Restaurants_reduced.loc[:,['business_id','text']]

reviews_grouped_by_business['reviews'] = reviews_grouped_by_business.groupby(['business_id'])['text'].transform(lambda x : ' | '.join(x))

# drop duplicate data
reviews_grouped_by_business = reviews_grouped_by_business.drop(['text'], axis=1)
reviews_grouped_by_business = reviews_grouped_by_business.drop_duplicates()
reviews_grouped_by_business.reset_index(inplace=True)
reviews_grouped_by_business.shape


# In[6]:


#Vectorize reviews
count_vectorizer = CountVectorizer(min_df = .01,max_df = .99, tokenizer = WordPunctTokenizer().tokenize)
reviews_grouped_by_business = reviews_grouped_by_business.fillna('')
vectorized_reviews = count_vectorizer.fit_transform(reviews_grouped_by_business['reviews'])
vectorized_reviews.shape


# # 3. Vectorize Categories

# In[7]:


df_categories = df_City_Restaurants.loc[:,['business_id','categories']]
#Vectorize categories
cat_count_vectorizer = CountVectorizer(min_df = .01,max_df = .99, tokenizer = WordPunctTokenizer().tokenize)
df_categories = df_categories.fillna('')
vectorized_categories = cat_count_vectorizer.fit_transform(df_categories['categories'])
vectorized_categories.shape


# # 4. Run Evaluation on the Complete List of Restaurants

# In[8]:


index_list = list(range(0, df_City_Restaurants.shape[0]-1))


# # 5. Evaluation Based on Reviews

# In[9]:


hit_rate_list = []
similarity_list = []
star_rating_list = []
review_count_list = []

for random_seed in index_list:
    #Find all the seed restaurant's reviews
    seed_restaurant_id = df_City_Restaurants.iloc[random_seed,0]
    seed_restaurant_category = df_City_Restaurants.loc[random_seed:random_seed,['categories']]
    selected_reviews = reviews_City_Restaurants_reduced[reviews_City_Restaurants_reduced['business_id']==seed_restaurant_id]
    selected_reviews = selected_reviews['text']
    df_City_Restaurants.iloc[random_seed]
    #Find most similar reviews to the reviews of seed restaurant using cosine similarity
    distance = cdist(count_vectorizer.transform(selected_reviews).todense().mean(axis=0), 
                  vectorized_reviews.todense(),metric='cosine')
    #Remove the seed restaurant from the list of recommendations
    if (df_City_Restaurants.iloc[random_seed,0] in reviews_grouped_by_business['business_id'].unique()) == True:
        #Find seed restaurant to remove it
        to_remove = reviews_grouped_by_business[reviews_grouped_by_business['business_id']==df_City_Restaurants.iloc[random_seed,0]].index[0]
        #Find the 10 most similar restaurants based on recommendations
        most_similar = distance.argsort().ravel()[:11]
        most_similar = most_similar[most_similar != to_remove]
        most_similar = most_similar[:10]
    else:
        most_similar = distance.argsort().ravel()[:10]

    #Display the 10 most similar restaurants based on recommendations
    df_most_similar = df_City_Restaurants.loc[df_City_Restaurants['business_id'].isin(reviews_grouped_by_business['business_id'].iloc[most_similar]), ['business_id', 'categories', 'name', 'stars','review_count']]
    #df_most_similar
    
    #Find all users who reviewed seed restaurant
    reviewed_seed_restaurant = reviews_City_Restaurants_reduced[reviews_City_Restaurants_reduced['business_id']== seed_restaurant_id]
    #Find all reviews by users who reviewed seed restaurant
    reviews_to_check = reviews_City_Restaurants_reduced[reviews_City_Restaurants_reduced['user_id'].isin(reviewed_seed_restaurant['user_id'])]

    #Calculate hit rate (or recall)
    #A hit is achieved if recommended restaurant was reviewed by at least one user who also reviewed seed restaurant    
    hit_count = 0
    try_count = 0
    for i in df_most_similar['business_id']:
        if (i in reviews_to_check['business_id'].unique()) == True:
            hit_count = hit_count +1
        try_count = try_count +1
    hit_rate = hit_count/try_count
    hit_rate_list.append(hit_rate)
    
    #Calculate similarity (or diversity)
    total_similarity = 0
    try_count = 0
    for a, b in itertools.combinations(df_most_similar['categories'], 2):
        total_similarity = total_similarity + string_similarity(a, b)
        try_count = try_count + 1
    similarity_rate = total_similarity/try_count
    similarity_list.append(similarity_rate)
    
    #Calculate average star rating  
    star_rating_list.append(sum(df_most_similar['stars']) / 10)
    
    #Calculate average number of reviews (or popularity)
    review_count_list.append(sum(df_most_similar['review_count']) / 10)


# In[10]:


average_hit_rate = sum(hit_rate_list) / len(hit_rate_list)
average_similarity = sum(similarity_list) / len(similarity_list)
average_star_rating = sum(star_rating_list) / len(star_rating_list)
average_review_count = sum(review_count_list) / len(review_count_list)

print("Average hit rate is: ",str(average_hit_rate))
print("Average diversity is: ",str(1-average_similarity))
print("Average star rating is: ",str(average_star_rating))
print("Average review count is: ",str(average_review_count))


# # 6. Evaluation Based on Categories

# In[11]:


hit_rate_list_cat = []
similarity_list_cat = []
star_rating_list_cat = []
review_count_list_cat = []

for random_seed in index_list:
    seed_restaurant_category = df_City_Restaurants.loc[random_seed:random_seed,['categories']]
    #Find most similar categories to the category of seed restaurant using cosine similarity
    distance = cdist(cat_count_vectorizer.transform(seed_restaurant_category['categories']).todense().mean(axis=0), 
                  vectorized_categories.todense(),metric='cosine')

    most_similar_cat = distance.argsort().ravel()[:10]

    #Display the 10 most similar restaurants based on recommendations
    df_most_similar_cat = df_City_Restaurants.loc[df_City_Restaurants['business_id'].isin(df_categories['business_id'].iloc[most_similar_cat]), ['business_id', 'categories', 'name', 'stars','review_count']]
    
    #Find all users who reviewed seed restaurant
    seed_restaurant_id = df_City_Restaurants.iloc[random_seed,0]
    reviewed_seed_restaurant = reviews_City_Restaurants_reduced[reviews_City_Restaurants_reduced['business_id']== seed_restaurant_id]
    #Find all reviews by users who reviewed seed restaurant
    reviews_to_check = reviews_City_Restaurants_reduced[reviews_City_Restaurants_reduced['user_id'].isin(reviewed_seed_restaurant['user_id'])]

    #Calculate hit rate (or recall)
    #A hit is achieved if recommended restaurant was reviewed by at least one user who also reviewed seed restaurant
    hit_count = 0
    try_count = 0
    for i in df_most_similar_cat['business_id']:
        if (i in reviews_to_check['business_id'].unique()) == True:
            hit_count = hit_count +1
        try_count = try_count +1
    hit_rate = hit_count/try_count
    hit_rate_list_cat.append(hit_rate)
       
    #Calculate similarity (or Diversity)
    total_similarity = 0
    try_count = 0
    for a, b in itertools.combinations(df_most_similar_cat['categories'], 2):
        total_similarity = total_similarity + string_similarity(a, b)
        try_count = try_count + 1
    similarity_rate_cat = total_similarity/try_count
    similarity_list_cat.append(similarity_rate_cat)
    
    #Calculate average star rating  
    star_rating_list_cat.append(sum(df_most_similar_cat['stars']) / 10)
    
    #Calculate average number of reviews (or popularity)
    review_count_list_cat.append(sum(df_most_similar_cat['review_count']) / 10)


# In[12]:


average_hit_rate_cat = sum(hit_rate_list_cat) / len(hit_rate_list_cat)
average_similarity_cat = sum(similarity_list_cat) / len(similarity_list_cat)
average_star_rating_cat = sum(star_rating_list_cat) / len(star_rating_list_cat)
average_review_count_cat = sum(review_count_list_cat) / len(review_count_list_cat)

print("Average hit rate using categories is: ",str(average_hit_rate_cat))
print("Average diversity using categories is: ",str(1-average_similarity_cat))
print("Average star rating using categories is: ",str(average_star_rating_cat))
print("Average review count using categories is: ",str(average_review_count_cat))


# # 7. Evaluation Based on Randomly Selecting 10 Restaurants

# In[13]:


hit_rate_list_random = []
similarity_list_random = []
star_rating_list_random = []
review_count_list_random = []

for random_seed in index_list:
    df_random_recommendations = df_City_Restaurants.sample(n = 10)
    
    #Find all users who reviewed seed restaurant
    seed_restaurant_id = df_City_Restaurants.iloc[random_seed,0]
    reviewed_seed_restaurant = reviews_City_Restaurants_reduced[reviews_City_Restaurants_reduced['business_id']== seed_restaurant_id]
    #Find all reviews by users who reviewed seed restaurant
    reviews_to_check = reviews_City_Restaurants_reduced[reviews_City_Restaurants_reduced['user_id'].isin(reviewed_seed_restaurant['user_id'])]

   
    #Calculate hit rate (or recall)
    #A hit is achieved if recommended restaurant was reviewed by at least one user who also reviewed seed restaurant    
    hit_count = 0
    try_count = 0
    for i in df_random_recommendations['business_id']:
        if (i in reviews_to_check['business_id'].unique()) == True:
            hit_count = hit_count +1
        try_count = try_count +1
    hit_rate = hit_count/try_count
    hit_rate_list_random.append(hit_rate)
    
    #Calculate similarity (or Diversity)
    total_similarity = 0
    try_count = 0
    for a, b in itertools.combinations(df_random_recommendations['categories'], 2):
        total_similarity = total_similarity + string_similarity(a, b)
        try_count = try_count + 1
    similarity_rate_random = total_similarity/try_count
    similarity_list_random.append(similarity_rate_random)
    
    #Calculate average star rating  
    star_rating_list_random.append(sum(df_random_recommendations['stars']) / 10)
    
    #Calculate average number of reviews (or popularity)
    review_count_list_random.append(sum(df_random_recommendations['review_count']) / 10)    


# In[14]:


average_hit_rate_random = sum(hit_rate_list_random) / len(hit_rate_list_random)
average_similarity_random = sum(similarity_list_random) / len(similarity_list_random)
average_star_rating_random = sum(star_rating_list_random) / len(star_rating_list_random)
average_review_count_random = sum(review_count_list_random) / len(review_count_list_random)

print("Average hit rate using random recommendations is: ",str(average_hit_rate_random))
print("Average diversity using random recommendations is: ",str(1-average_similarity_random))
print("Average star rating using random recommendations is: ",str(average_star_rating_random))
print("Average review count using random recommendations is: ",str(average_review_count_random))


# In[15]:


#Calculate average star rating for the whole sample
total_average_star_rating = sum(df_City_Restaurants['stars']) / len(df_City_Restaurants['review_count'])
total_average_star_rating


# In[16]:


#Calculate average review count for the whole sample
total_average_review_count = sum(df_City_Restaurants['review_count']) / len(df_City_Restaurants['review_count'])
total_average_review_count


# # 8. Visualize Evaluation Metrics

# In[33]:


metrics_data = {'Model Name':['Recommender','Baseline_1 Categories','Baseline_2 Random'],
                'Hit Rate':[average_hit_rate, average_hit_rate_cat, average_hit_rate_random],
                'Diversity':[1-average_similarity, 1-average_similarity_cat, 1-average_similarity_random],
                'Avg Star Rating':[average_star_rating, average_star_rating_cat, average_star_rating_random],
                'Avg Review Count':[average_review_count, average_review_count_cat, average_review_count_random]}
    
df_metrics = pd.DataFrame(data = metrics_data)
df_metrics


# In[34]:


#Plot Hit Rate
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

ax = sns.barplot(data=df_metrics, x = 'Model Name', y='Hit Rate')
ax.bar_label(ax.containers[0])


# In[35]:


#Plot Diversity
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

ax = sns.barplot(data=df_metrics, x = 'Model Name', y='Diversity')
ax.bar_label(ax.containers[0])


# In[36]:


#Plot Aervage Star Rating
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

ax = sns.barplot(data=df_metrics, x = 'Model Name', y='Avg Star Rating')
ax.bar_label(ax.containers[0])


# In[37]:


#Plot Aervage Review Count (Popularity)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

ax = sns.barplot(data=df_metrics, x = 'Model Name', y='Avg Review Count')
ax.bar_label(ax.containers[0])

