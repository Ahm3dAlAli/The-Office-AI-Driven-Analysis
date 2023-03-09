#!/usr/bin/env python
# coding: utf-8

# # Machine Learning in Python - Group Project 1
# 
# **Due Friday, March 10th by 16.00 pm.**
# 
# *include contributors names here (such as Name1, Name2, ...)*

# ## General Setup

# In[1]:


# Add any additional libraries or submodules below

# Data libraries
import numpy as np
import pandas as pd
import random

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# Plotting defaults
plt.rcParams['figure.figsize'] = (8,5)
plt.rcParams['figure.dpi'] = 80

#NLP Libraries 
import re 
import string
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from textblob import TextBlob


# sklearn modules that are necessary
import sklearn
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# Text egenration 
import gpt_2_simple as gpt2 
import os
import re 

## 1. Introduction

*This section should include a brief introduction to the task and the data (assume this is a report you are delivering to a client).* 
### 1.1 Background Info and Problem Statemnt 

The Office is an American television show that aired for nine seasons from 2005 to 2013. The show follows the daily lives of employees at the Scranton, Pennsylvania branch of the fictional Dunder Mifflin Paper Company. Over the years, The Office has gained a significant following and remains popular even after the show ended. As a result, NBC Universal has expressed interest in producing a special reunion episode of the show. To ensure the reunion episode's success, NBC Universal has hired a data science contractor to help them better understand what made some episodes more popular than others. The contractor's task is to develop a reliable and accurate predictive model that captures the underlying relationships between different features of The Office and audience ratings. The insights gained from this model will then be used to advise NBC Universal on how to produce the highest rated reunion episode possible. We aim to determine genre of the Reunion Episode is it going to be comedic, mysterious thriller, will they discover some secrets in "The Office", should it be a seasonal episode speical where they re-union can this lead to a higher rating ?, main characters to inlude , duration of episode, director and writers to produce the episode .  



### 1.2 Data Avialbility and Usage 

The available data includes information such as season and episode numbers, episode names, director and writer information, IMDb ratings, total votes, air dates, and character information. Additionally, external data sources such as transcript dialouge has season and episode  numbers, titles , scene , speaker , line .
The IMDB rating dataset was given and teh transciprt data was found from reddit datasets https://www.reddit.com/r/datasets/comments/6yt3og/every_line_from_every_episode_of_the_office_us/ .


### 1.3 Pipeline of ourn Work

1. Gather Data
    - Data of IMDB Ratings of The Office Show 
    - Dialoge Transcripts of The Office Show

The Office IMDb dataset contains:-

    - season : Season Number ( 1 to 9 )
    - episode : Episode number
    - episode_name : Name of the episode
    - imdb_rating : Ratings given to the episode on IMDb
    - total_votes : Votes given to the episode on IMDb
    - air_date : Release date of the episode
    - director : Director(s) of the episode
    - writers : Writer(s) of the episode
    - main_chars : Main actors in episode
    - n_lines : Number of script lines in episode 
    - n_words : Number of words in episode
    - n_directions : Number of directions given by directors
    - n_speak_char : Average Number of characters per Line spoken

The Office transcripts contains:-

    - season : Season number
    - episode : Episode number
    - scene : Scene number
    - speaker : Actor in the scene
    - line_text : Lines of the speaker
    - deleted: Indicator of whether scene is deleted 

2. Data Cleaning and preparation 
    - Clean  the IMDb dataset, we can remove any missing or duplicate values and convert the data types as necessary
    - Clean the transcript data by removing stop words, stemming or lemmatizing words, and converting all words to lowercase.
    - Merge the transcript data with the IMDb ratings data by episode name to create a single dataset.
    - Clean and preprocess the merged dataset by removing missing values, duplicates, and irrelevant columns.
    - Extract additional features from the data, such as the number of lines, words, and characters per scene, the number of directions, the season, and the air date of each episode.

3. EDA
    A. IMDB Ratings
        - Metadata
        - Summary Statstics 
        - Directors, Writers and Actors
        - Identify which variables are most strongly correlated with ratings
            I. Grouped by Season
            II. Grouped by Directors
            III. Grouped by Writers 
            IV.  Grouped by Actor
            V.  Grouped by duration of episode 

    B. Transcript
        - Metadata
        - Summary Statstics 
        - Interactions and sentimental analysis  
            I. Who spoke the most of the actors 
            II. Interactions Between Actors ,Who speak to Whom? inspired by https://www.reddit.com/user/Gandagorn
            III. What interacions leads to more rating  
            IV. Highest rating by scene number

    

4. Pre-proccessing Feature Engineering  
    - Engineer additional features from the data, such as sentiment scores, character interaction metrics, and episode themes.
    -Consider using techniques such as PCA to reduce dimenisonality, and preserve varibility of data for different modeling purposes .



5. Methodology
    -Baseline Models, Supervised learning Model Predicitve Modeling for the rating . Modleing Inlfuence of varaibles on hihger rarting , modelling what settings of an epsiode influences a higher rating, and what directors, writers and actors leads to a hgiher rated episode.
    -Exepriemntal Set-up, Train various models on the prepared and engineered dataset to predict IMDb ratings, such as linear regression and SVM. Evaluate the performance of each model using cross-validation and metrics such as R-squared. Choose the best-performing model based on the evaluation metrics.
    -Hyper-parameter tuning. 


6. Results and Analysis 
    -Model Comparision , Interpret the final model by examining the most important features and their impact on the predicted ratings. Validate the model using out-of-sample testing to ensure its reliability and generalization.
    -Result and Analysis, Use the model to make predictions about the reunion episode's potential ratings based on different scenarios, such as different genres, episode durations, and main characters.Provide insights and recommendations to NBC Universal based on the model's predictions and interpretation.

7. Application 
    -Based on trained data and outcomes  Based on the reuslts we got for the most suitable setting assocaited with high ratings, generate a new episode script using GPT-2, inspired by https://github.com/minimaxir/gpt-2-simple and https://towardsdatascience.com/text-generation-with-python-and-gpt-2-1fecbff1635b






# In[2]:


#########################################
# Cleaning and preapring rating dataset #
#########################################


imdb_data = pd.read_csv("/Users/ahmed/Documents/UOE/Courses/Semester 2/Machine learning in python /Projects/The Office Rating and Trnascript /the_office.csv")

# Drop any rows with missing values
imdb_data.dropna(inplace=True)

# Convert the numeric comuns to float 
imdb_data["imdb_rating"] = imdb_data["imdb_rating"].astype(float)
imdb_data["total_votes"] = imdb_data["total_votes"].astype(float)
imdb_data['n_lines'] = imdb_data["n_lines"].astype(float)
imdb_data['n_directions'] = imdb_data["n_directions"].astype(float)
imdb_data['n_words'] = imdb_data["n_words"].astype(float)
imdb_data['n_speak_char'] = imdb_data["n_speak_char"].astype(float)

# Convert the Date column to a datetime object
imdb_data["air_date"] = pd.to_datetime(imdb_data["air_date"])

# Extract the year from the Date column
imdb_data["Year"] = (imdb_data["air_date"].dt.year).astype(int)

# Drop the Date column
imdb_data.drop("air_date", axis=1, inplace=True)


# Remove any whitespace from the Director, Writers and main characters columns
imdb_data["director"] = imdb_data["director"].str.strip().str.split(';')
imdb_data["writer"] = imdb_data["writer"].str.strip().str.split(';')
imdb_data['main_chars'] =imdb_data["main_chars"].str.strip().str.split(';')

# Check the cleaned data
print(imdb_data.head())


############################################
# Cleaning and preapring trancript dataset #
############################################
# read the transcript data from a CSV file
transcript = pd.read_csv('/Users/ahmed/Documents/UOE/Courses/Semester 2/Machine learning in python /Projects/The Office Rating and Trnascript /the-office-lines - scripts.csv')

# Drop any rows with missing values
transcript.dropna(inplace=True)

# remove deleted scenes
transcript = transcript[transcript['deleted'] == False]

# drop the 'deleted' column
transcript = transcript.drop(columns=['deleted'])

# Include those lines for main characters only found in the imdb data set
imdb_exploded = imdb_data.explode('main_chars')
unique_chars = imdb_exploded['main_chars'].unique()
transcript[transcript['speaker'].isin(unique_chars)]



def clean_transcript(line_text):
    # convert to lowercase
    transcript = line_text.lower()
    # remove stop words
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    transcript = ' '.join([word for word in transcript.split() if word not in stop_words])
    # remove special characters and digits
    transcript = re.sub(r"[^a-zA-Z\s]+", "", transcript)
    # convert to lowercase
    transcript = transcript.lower()
    # remove leading/trailing whitespaces
    transcript = transcript.strip()
    # replace multiple spaces with a single space
    transcript = re.sub(r"\s+", " ", transcript)
    return transcript

transcript_cleaned=transcript
transcript_cleaned['line_text'] = transcript_cleaned['line_text'].apply(clean_transcript)

#Check the cleaned data
print(transcript_cleaned.head())


# In[3]:


#Grouping Transcript Data per Season and Epsiode 

transcript_grouped = transcript_cleaned.groupby(["season", "episode"]).agg({ "scene": "count", "line_text": lambda x: " ".join(x)}).reset_index()
transcript_grouped.rename(columns={"scene": "num_scenes", "line_text": "transcript"}, inplace=True)

merged_data = pd.merge(transcript_grouped,imdb_data, on=['season', 'episode'])

sns.pairplot(
    data=merged_data, 
    aspect=.85,
    hue='season');


# In[4]:




# Meta data 
print(transcript.info())

# Display summary statistics
print(transcript.describe())

#Who spoke most out of the actors 
import matplotlib.pyplot as plt

# Get the top 10 characters based on the number of lines they spoke
top_characters = transcript.groupby('speaker').size().sort_values(ascending=False)[:10]

# Plot a bar chart of the top characters
plt.bar(top_characters.index, top_characters.values)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Character')
plt.ylabel('Number of Lines Spoken')
plt.show()

# Directors and Avg. Episode Rating 

def dir_mean_rating(director, imdb_data):
    mask = []
    for i, row in imdb_data.iterrows():
        if director in row["director"]:
            mask.append(i)
            
    return imdb_data.loc[mask, "imdb_rating"].mean()

list1 = []
for x in imdb_data["director"].explode().value_counts().head(11).keys():
    list1.append(dir_mean_rating(x, imdb_data))

xs = imdb_data["director"].explode().value_counts().head(11).keys()
ys = list1

g = sns.barplot(x=xs,y=ys)
g.set(ylim=(7.5,9))
g.set_xticklabels(g.get_xticklabels(), rotation=90)
plt.xlabel("Directors in descending order of the no. of shows they directed")
plt.ylabel("Average rating of their shows")
plt.title("Directors and their episodes' ratings")
plt.show()



# Directors and Avg. Episode Rating 

def wrot_mean_rating(writer, imdb_data):
    mask = []
    for i, row in imdb_data.iterrows():
        if writer in row["writer"]:
            mask.append(i)
            
    return imdb_data.loc[mask, "imdb_rating"].mean()

list1 = []
for x in imdb_data["writer"].explode().value_counts().head(11).keys():
    list1.append(dir_mean_rating(x, imdb_data))

xs = imdb_data["writer"].explode().value_counts().head(11).keys()
ys = list1

g = sns.barplot(x=xs,y=ys)
g.set(ylim=(7.5,9))
g.set_xticklabels(g.get_xticklabels(), rotation=90)
plt.xlabel("Wroter in descending order of the no. of shows they wrote")
plt.ylabel("Average rating of their shows")
plt.title("Writers and their episodes' ratings")
plt.show()


#Interactions Between Actors, Who spoke to Whom

# Get 20 characters
main_characters = list(transcript['speaker'].value_counts().index[:20])

# Create networkx object
G = nx.DiGraph()

# Get conversation info betwwen characters
scene_before = ""
episode_id_before = -1
for i in range(len(transcript)):
    # Check if episode and location of text is the same
    if scene_before != transcript["scene"].iloc[i] or transcript["episode"].iloc[i] != episode_id_before:
        scene_before = transcript.iloc[i]["scene"]
        episode_id_before = transcript.iloc[i]["episode"]
        continue
    scene_before = transcript.iloc[i]["scene"]
    episode_id_before = transcript.iloc[i]["episode"]
    # Get characters
    character1 = transcript["speaker"].iloc[i]
    character2 = transcript["speaker"].iloc[i+1]
    # Fail check for character not in the interested list
    if character1 not in main_characters or character2 not in main_characters:
        continue

    sorted_characters = (character1, character2)
    try:
        # Add +1 to weight if characters have conversation on the same sence
        G.edges[sorted_characters]["weight"] += 1
    except KeyError:
        G.add_edge(sorted_characters[0], sorted_characters[1], weight=1)

# Plot the graph
plt.figure(figsize=(40, 40))
pos = nx.circular_layout(G)

edges = G.edges()

################
colors = [G[u][v]['weight']**0.09 for u, v in edges]

weights = [G[u][v]['weight']**0.45 for u, v in edges]

cmap = plt.cm.get_cmap('Blues')

nx.draw_networkx(G, pos, width=weights, edge_color=colors,
                 node_color="white", edge_cmap=cmap, with_labels=False)

labels_pos = {name: [pos_list[0], pos_list[1]-0.04] for name, pos_list in pos.items()}
nd = nx.draw_networkx_labels(G, labels_pos, font_size=50, font_family="sans-serif",
                             font_color="#000000", font_weight='bold')

ax = plt.gca()
ax.collections[0].set_edgecolor('#000000')
ax.margins(0.25)
plt.show()



#Highest rating by scene number, we can group the merged data by scene number and calculate the maximum rating for each scene.
# Group by scene number and calculate max rating for each scene



max_ratings_by_scene = merged_data.groupby('num_scenes')['imdb_rating'].max().reset_index()

# Plot highest rating by scene number
plt.figure(figsize=(10,6))
sns.lineplot(data=max_ratings_by_scene, x='num_scenes', y='imdb_rating')
plt.title('Highest Rating by Scene Number')
plt.xlabel('Scene Number')
plt.ylabel('Rating')
plt.show()

#Highest rating by scene number, we can group the merged data by scene number and calculate the maximum rating for each scene.
# Group by scene number and calculate max rating for each scene



max_ratings_by_numspeakers = merged_data.groupby('n_speak_char')['imdb_rating'].max().reset_index()

# Plot highest rating by scene number
plt.figure(figsize=(10,6))
sns.lineplot(data=max_ratings_by_numspeakers, x='n_speak_char', y='imdb_rating')
plt.title('Highest Rating by Number of Actors')
plt.xlabel('Actors')
plt.ylabel('Rating')
plt.show()

#Number of Directors 
merged_data['n_directors'] = merged_data['director'].apply(lambda x: len(x))
max_ratings_by_directornum = merged_data.groupby('n_directors')['imdb_rating'].max().reset_index()

# Plot highest rating by scene number
plt.figure(figsize=(10,6))
sns.lineplot(data=max_ratings_by_directornum, x='n_directors', y='imdb_rating')
plt.title('Highest Rating by Number of Directors')
plt.xlabel('Directors')
plt.ylabel('Rating')
plt.show()


#Number of writers
merged_data['n_writers'] = merged_data['writer'].apply(lambda x: len(x))
max_ratings_by_writernum = merged_data.groupby('n_writers')['imdb_rating'].max().reset_index()

# Plot highest rating by scene number
plt.figure(figsize=(10,6))
sns.lineplot(data=max_ratings_by_writernum, x='n_writers', y='imdb_rating')
plt.title('Highest Rating by Number of Writers')
plt.xlabel('Writers')
plt.ylabel('Rating')
plt.show()




#Positive and negative sentiments per episode and then visualize rating for seasons as positive, negative or neutral sentiment, we need to perform sentiment analysis on the lines spoken by each character in the transcript data. We can use the TextBlob library to do this.

# Calculate sentiment polarity for each line of dialogue
merged_data['polarity'] = merged_data['transcript'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Group by episode and calculate average polarity for each episode

# Categorize episodes as positive, negative or neutral based on average polarity
merged_data['sentiment'] = np.where(merged_data['polarity'] > 0.1, 'positive',
                                               np.where(merged_data['polarity'] < 0, 'negative', 'neutral'))



# Plot season ratings by sentiment
plt.figure(figsize=(10,6))
sns.boxplot(data=merged_data, x='season', y='imdb_rating', hue='sentiment')
plt.title('Season Ratings by Sentiment')
plt.xlabel('Season')
plt.ylabel('Rating')
plt.show()


#Plot duration versus rating 


sns.boxplot(data=merged_data, x=(merged_data["n_lines"]/(merged_data["n_speak_char"]))+(((merged_data["n_directions"])/merged_data["main_chars"].str.len()))+np.mean(merged_data["n_speak_char"]/merged_data["main_chars"].str.len()),y='imdb_rating')
plt.figure(figsize=(10,6))
plt.title('Avg Rating by Duration')
plt.xlabel('Duration')
plt.ylabel('Rating')
plt.show()




# In[5]:



######################################
# Feature Engineering Merged Dataset #
######################################

#Merge Data sets
imdb_data_merged = pd.merge(transcript_grouped,imdb_data, on=['season', 'episode'])


# Continious Variables Piepline
###############################

#Number of Actors
imdb_data_merged=imdb_data_merged.rename(columns={'n_speak_char':'n_actors'})

#Number of Directors 
imdb_data_merged['n_directors'] = imdb_data_merged['director'].apply(lambda x: len(x))
#Number of writers
imdb_data_merged['n_writers'] = imdb_data_merged['writer'].apply(lambda x: len(x))

# Duration of episode 
imdb_data_merged["duration"]=(imdb_data["n_lines"]/(imdb_data["n_speak_char"]))+(((imdb_data["n_directions"])/imdb_data["main_chars"].str.len()))+np.mean(imdb_data["n_speak_char"]/imdb_data["main_chars"].str.len())

# define a function to calculate sentiment scores for each line of dialogue
def get_sentiment_score(line):
    blob = TextBlob(line)
    sentiment = blob.sentiment.polarity
    return sentiment

# apply the sentiment score function to the transcript data
imdb_data_merged['sentiment_score'] = imdb_data_merged['transcript'].apply(get_sentiment_score)


# Adding seaosnal Episodes ,https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwig57qXpcb9AhWFUsAKHa4_AaYQFnoECA0QAQ&url=https%3A%2F%2Fscreenrant.com%2Fthe-office-holiday-episodes-ranked-christmas-thanksgiving%2F&usg=AOvVaw3Hgc1TPXreGVyGskWVNqk6
Holiday_Episode=[[7,9],[6,22],[8,10],[5,11],[6,19],[3,6],[5,18],[9,9],[7,11],[7,12],[7,6],[3,10],[3,11],[2,5],[6,13],[2,16],[2,10]]

def is_holiday_episode(row):
    if [row['season'], row['episode']] in Holiday_Episode:
        return 1
    else:
        return 0

imdb_data_merged['Holiday_Episode'] = imdb_data_merged.apply(is_holiday_episode, axis=1)


#Multi Part Episodes 
#imdb_data_merged[imdb_data_merged['episode_name'].str.contains('Part 1|Part 2|Parts 1&2')]
Multi_Part_Episode=[[3,10],[4,1],[4,3],[4,5],[4,7],[5,1],[5,14],[5,16],[5,17],[6,4],[6,17],[6,11]]

def is_multipart_episode(row):
    if [row['season'], row['episode']] in Multi_Part_Episode:
        return 1
    else:
        return 0

imdb_data_merged['MultiPart_Episode'] = imdb_data_merged.apply(is_multipart_episode, axis=1)

#Standarize nuimerical features 
from sklearn.preprocessing import MinMaxScaler

# create a StandardScaler object
scaler = MinMaxScaler()

# select only the numerical columns to be standardized
num_cols = ['total_votes', 'n_lines', 'n_directions', 'n_words','sentiment_score']
# fit and transform the numerical columns using the scaler
imdb_data_merged[num_cols] = scaler.fit_transform(imdb_data_merged[num_cols])


# In[6]:




# Encode Categorical Predictor Varaibles
############################################
#directors
def add_dir_to_array(arr):
    return ["director." + a for a in arr]

imdb_data_merged["director"] = imdb_data_merged["director"].apply(add_dir_to_array)


mlb = MultiLabelBinarizer()

imdb_data_merged = imdb_data_merged.join(pd.DataFrame(mlb.fit_transform(imdb_data_merged['director']),columns=mlb.classes_))

#directors
def add_wri_to_array(arr):
    return ["writer." + a for a in arr]

imdb_data_merged["writer"] = imdb_data_merged["writer"].apply(add_wri_to_array)


mlb = MultiLabelBinarizer()


imdb_data_merged = imdb_data_merged.join(pd.DataFrame(mlb.fit_transform(imdb_data_merged['writer']),columns=mlb.classes_))

#actors
def add_act_to_array(arr):
    return ["actor." + a for a in arr]

imdb_data_merged["main_chars"] = imdb_data_merged["main_chars"].apply(add_act_to_array)


mlb = MultiLabelBinarizer()


imdb_data_merged = imdb_data_merged.join(pd.DataFrame(mlb.fit_transform(imdb_data_merged['main_chars']),columns=mlb.classes_))


# Two differen Data set for PCA modeling later on 
imdb_data_settings=imdb_data_merged
imdb_data_act_writ_dire=imdb_data_merged
imdb_data_rating= imdb_data_merged
# Based on intial EDA we decided to drop 
imdb_data_act_writ_dire = imdb_data_act_writ_dire.drop(['episode_name','season','episode', 'director','writer','main_chars','transcript','n_words','n_lines'], axis=1)

imdb_data_settings = imdb_data_settings.drop(['episode_name','season','episode', 'director','writer','main_chars','transcript','n_words','n_lines'], axis=1)


# In[7]:



# PCA Dimensionality Reduction for Director,Writer and Actor Column, follow after first categroical tranfromation pipeline
###########################################################################################################################

PCA_Data = imdb_data_settings.filter(regex="^(director\.|writer\.|actor\.)")

#Directors
director_data = PCA_Data.filter(like='director.')
# normalize the mean of features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(director_data)

# scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_normalized)

director_data=X_scaled


# apply PCA
pca = PCA()
pca.fit(director_data)

# tune PCA using grid search
params = {'n_components': [1,2,3,4,5,6,7,8,9,10]}
grid_pca = GridSearchCV(pca, params)
grid_pca.fit(director_data)
pca_best = grid_pca.best_estimator_
X_pca_best = pca_best.transform(director_data)

# print evaluation scores for each combination of hyperparameters
results = grid_pca.cv_results_
for i in range(len(results['params'])):
    print("Hyperparameters: ", results['params'][i])
    print("Mean score: ", results['mean_test_score'][i])
    print("Standard deviation: ", results['std_test_score'][i])
    print("\n")
 


#Get the explained variance ratio for each component
variance_ratio = pca_best.explained_variance_ratio_

# Create a bar plot of the explained variance ratios
plt.bar(range(1, len(variance_ratio) + 1), variance_ratio)

# Add labels and title
plt.xlabel('PCA Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio for Directors PCA Components')

# Show the plot
plt.show()


print('Optimal Number of Componenets :',pca_best)


X_pca_best = pd.DataFrame(X_pca_best, columns=['PCA_Directors'+str(i+1) for i in range(pca_best.n_components)])

imdb_data_Modeling_settings = imdb_data_settings.filter(regex="^(?!director\.|writer\.|actor\.)").join(X_pca_best)





# Writers
writers_data = PCA_Data.filter(like='writer.')
# normalize the mean of features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(writers_data)

# scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_normalized)

writers_data=X_scaled

# apply PCA
pca = PCA()
pca.fit(writers_data)

# tune PCA using grid search
params = {'n_components': [1,2,3,4,5,6,7,8,9,10]}
grid_pca = GridSearchCV(pca, params)
grid_pca.fit(writers_data)
pca_best = grid_pca.best_estimator_
X_pca_best = pca_best.transform(writers_data)

# print evaluation scores for each combination of hyperparameters
results = grid_pca.cv_results_
for i in range(len(results['params'])):
    print("Hyperparameters: ", results['params'][i])
    print("Mean score: ", results['mean_test_score'][i])
    print("Standard deviation: ", results['std_test_score'][i])
    print("\n")



# Get the explained variance ratio for each component
variance_ratio = pca_best.explained_variance_ratio_

# Create a bar plot of the explained variance ratios
plt.bar(range(1, len(variance_ratio) + 1), variance_ratio)

# Add labels and title
plt.xlabel('PCA Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio for Writers PCA Components')

# Show the plot
plt.show()

print('Optimal Number of Componenets :',pca_best)
X_pca_best = pd.DataFrame(X_pca_best, columns=['PCA_Writer'+str(i+1) for i in range(pca_best.n_components)])

imdb_data_Modeling_settings = imdb_data_Modeling_settings.join(X_pca_best)


# Actors
actors_data = PCA_Data.filter(like='actor.')
# normalize the mean of features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(actors_data)

# scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_normalized)

actors_data=X_scaled

# apply PCA
pca = PCA()
pca.fit(actors_data)

# tune PCA using grid search
params = {'n_components': [1,2,3,4,5,6,7,8,9,10]}
grid_pca = GridSearchCV(pca, params)
grid_pca.fit(actors_data)
pca_best = grid_pca.best_estimator_
X_pca_best = pca_best.transform(actors_data)

# print evaluation scores for each combination of hyperparameters
results = grid_pca.cv_results_
for i in range(len(results['params'])):
    print("Hyperparameters: ", results['params'][i])
    print("Mean score: ", results['mean_test_score'][i])
    print("Standard deviation: ", results['std_test_score'][i])
    print("\n")



# Get the explained variance ratio for each component
variance_ratio = pca_best.explained_variance_ratio_

# Create a bar plot of the explained variance ratios
plt.bar(range(1, len(variance_ratio) + 1), variance_ratio)

# Add labels and title
plt.xlabel('PCA Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio for Actors PCA Components')

# Show the plot
plt.show()

print('Optimal Number of Componenets :',pca_best)
X_pca_best = pd.DataFrame(X_pca_best, columns=['PCA_Actors'+str(i+1) for i in range(pca_best.n_components)])

imdb_data_Modeling_settings = imdb_data_Modeling_settings.join(X_pca_best)



# In[ ]:


# PCA Dimensionality Reduction for all except Director,Writer and Actor Column, follow after second dataset transfomraiton 
###########################################################################################################################

PCA_Data = imdb_data_act_writ_dire.filter(regex="^(?!director.|actor.|writer.).*")



# normalize the mean of features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(PCA_Data)

# scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_normalized)

PCA_Data=X_scaled


# apply PCA
pca = PCA()
pca.fit(PCA_Data)



# apply PCA
pca = PCA()
pca.fit(PCA_Data)

# tune PCA using grid search
params = {'n_components': [1,2,3,4,5,6,7,8,9,10]}
grid_pca = GridSearchCV(pca, params)
grid_pca.fit(PCA_Data)
pca_best = grid_pca.best_estimator_
X_pca_best = pca_best.transform(PCA_Data)

# print evaluation scores for each combination of hyperparameters
results = grid_pca.cv_results_
for i in range(len(results['params'])):
    print("Hyperparameters: ", results['params'][i])
    print("Mean score: ", results['mean_test_score'][i])
    print("Standard deviation: ", results['std_test_score'][i])
    print("\n")

#Get the explained variance ratio for each component
variance_ratio = pca_best.explained_variance_ratio_

# Create a bar plot of the explained variance ratios
plt.bar(range(1, len(variance_ratio) + 1), variance_ratio)

# Add labels and title
plt.xlabel('PCA Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio for Other Predictors  PCA Components')

# Show the plot
plt.show()


print('Optimal Number of Componenets :',pca_best)


X_pca_best = pd.DataFrame(X_pca_best, columns=['PCA_Setting_Predictors'+str(i+1) for i in range(pca_best.n_components)])


imdb_data_modeling_Int=imdb_data_act_writ_dire.filter(regex="^(director.|actor.|writer.|imdb_).*").join(X_pca_best)


# In[39]:


########################
# Modeling and tuning  #
########################

# Higher Rating #
###################

# Split the data into training and testing sets
train_data = imdb_data_Modeling_settings.sample(frac=0.7, random_state=42)
test_data = imdb_data_Modeling_settings.drop(train_data.index)


# Define the independent variables and the target variable
X_train = train_data.drop(['imdb_rating'], axis=1)
X_test = test_data.drop(['imdb_rating'], axis=1)
y_train = train_data['imdb_rating']
y_test = test_data['imdb_rating']







################################################################################
# Linear Regression#
####################

# Feature Selection #
#####################

# Define LassoCV model with 5-fold cross-validation
lasso_model = LassoCV(cv=5)

# Fit the model on the training data
lasso_model.fit(X_train, y_train)

# Print selected features and their coefficients
selected_features = X_train.columns[lasso_model.coef_ != 0]
print("Selected features:", selected_features)
print("Coefficients:", lasso_model.coef_[lasso_model.coef_ != 0])


# Plot feature coefficients
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(lasso_model.coef_, marker='o', linestyle='')

# Add feature names as tick labels
ax.set_xticklabels(X_train.columns, rotation=90)

# Add labels and title
ax.set_xlabel("Features")
ax.set_ylabel("Coefficient")
ax.set_title("Lasso Feature Coefficients")

plt.show()


# Define the independent variables and the target variable
X_train_LR= train_data.loc[:,['num_scenes', 'total_votes', 'n_actors', 'Year', 'duration',
       'PCA_Directors1', 'PCA_Directors2', 'PCA_Directors3', 'PCA_Directors8',
       'PCA_Writer1', 'PCA_Writer2', 'PCA_Actors1']]
X_test_LR = test_data.loc[:,['num_scenes', 'total_votes', 'n_actors', 'Year', 'duration',
       'PCA_Directors1', 'PCA_Directors2', 'PCA_Directors3', 'PCA_Directors8',
       'PCA_Writer1', 'PCA_Writer2', 'PCA_Actors1']]
y_train_LR = train_data['imdb_rating']
y_test_LR = test_data['imdb_rating']



# Define pipeline
linreg_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('linreg', LinearRegression())
])

# Define hyperparameters to tune
linreg_params = {
    'linreg__normalize': [True, False]
}

# Perform grid search cross-validation to find best hyperparameters
linreg_grid = GridSearchCV(linreg_pipe, linreg_params, cv=10, scoring='r2')
linreg_grid.fit(X_train_LR, y_train_LR)

# Train model with best hyperparameters
linreg_model = linreg_grid.best_estimator_
linreg_model.fit(X_train_LR, y_train_LR)

# Evaluate model performance
linreg_scores = cross_val_score(linreg_model, X_test_LR, y_test_LR, cv=10, scoring='r2')
print('Average R^2 score:', linreg_scores.mean() )

# Parameter choice 
print('Optimal Paraemter Choices :',linreg_model.get_params(deep=True))

#Model coefficents 
print('Model Coeffcients :',linreg_model.named_steps['linreg'].coef_)


# Plot feature coeff

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(linreg_model.named_steps['linreg'].coef_, marker='o', linestyle='')

ax.set_xticks(np.arange(12))
# Add feature names as tick labels
ax.set_xticklabels(['Number of scenes','Toal Votes','Number of Actors','Year of Air','Duration','PCA Director 1','PCA Directors 2', 'PCA Directors 3', 'PCA_Directors 8',
       'PCA Writer 1', 'PCA Writer 2', 'PCA Actors 1'], rotation=90)

# Add labels and title
ax.set_xlabel("Features")
ax.set_ylabel("Coefficient")
ax.set_title("Linear Regression Feature Coefficients")
ax.legend()

plt.show()

################################################################################################
# Random Forest #
#################

# Define hyperparameters to tune
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, 9],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 3, 4]
}

# Perform grid search cross-validation to find best hyperparameters
rf_grid = GridSearchCV(RandomForestRegressor(), rf_params, cv=10, scoring='r2')
rf_grid.fit(X_train, y_train)

# Train model with best hyperparameters
rf_model = rf_grid.best_estimator_
rf_model.fit(X_train, y_train)

# Evaluate model performance
rf_scores = cross_val_score(rf_model, X_test, y_test, cv=10, scoring='r2')
print('Random Forest R^2 scores:', rf_scores)
print('Average R^2 score:', rf_scores.mean())

# Get feature importances from the trained model
importances = rf_model.feature_importances_

# Get a list of feature names
feature_names = X_train.columns

# Create a dataframe with feature names and their importances
feature_importances = pd.DataFrame({'feature': feature_names, 'importance': importances})

# Sort features by importance (descending order)
feature_importances = feature_importances.sort_values('importance', ascending=False)

# Print the top 10 features and their importances
print(feature_importances.head(10))


# Parameter choice 
print('Optimal Paraemter Choices :',rf_model.get_params(deep=True))


# Plot the R2 scores for each method
plt.plot(linreg_scores, label='Linear Regression')
plt.plot(rf_scores, label='Random Forest Regressor')

# Add a title, x and y labels, and a legend
plt.title("R2 scores across all cross folds")
plt.xlabel("Cross fold")
plt.ylabel("R2 score")
plt.legend()

# Show the plot
plt.show()




# Plot feature importances
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(feature_importances['feature'], feature_importances['importance'], color='b')

# Add labels and title
ax.set_xlabel("Features")
ax.set_ylabel("Importance")
ax.set_title("Random Forest Regressor Feature Importance")

plt.xticks(rotation=90)
plt.show()


# In[20]:


# Setting #
###########
# Split the data into training and testing sets
train_data = imdb_data_Modeling_settings.sample(frac=0.7, random_state=42)
test_data = imdb_data_Modeling_settings.drop(train_data.index)

# Define the independent variables and the target variable
X_train = train_data.drop(['imdb_rating','Year','total_votes'], axis=1)
X_test = test_data.drop(['imdb_rating','Year','total_votes'], axis=1)
y_train = train_data['imdb_rating']
y_test = test_data['imdb_rating']




################################################################################
# Linear Regression#
####################

# Feature Selection #
#####################

# Define LassoCV model with 5-fold cross-validation
lasso_model = LassoCV(cv=5)

# Fit the model on the training data
lasso_model.fit(X_train, y_train)

# Print selected features and their coefficients
selected_features = X_train.columns[lasso_model.coef_ != 0]
print("Selected features:", selected_features)
print("Coefficients:", lasso_model.coef_[lasso_model.coef_ != 0])


# Plot feature coefficients
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(lasso_model.coef_, marker='o', linestyle='')

# Add feature names as tick labels
ax.set_xticklabels(X_train.columns, rotation=90)

# Add labels and title
ax.set_xlabel("Features")
ax.set_ylabel("Coefficient")
ax.set_title("Lasso Feature Coefficients")

plt.show()


# Define the independent variables and the target variable
X_train_LR= train_data.loc[:,['num_scenes', 'n_actors', 'duration', 'PCA_Directors1',
       'PCA_Directors2', 'PCA_Directors3', 'PCA_Actors1']]
X_test_LR = test_data.loc[:,['num_scenes', 'n_actors', 'duration', 'PCA_Directors1',
       'PCA_Directors2', 'PCA_Directors3', 'PCA_Actors1']]
y_train_LR = train_data['imdb_rating']
y_test_LR = test_data['imdb_rating']



# Define pipeline
linreg_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('linreg', LinearRegression())
])

# Define hyperparameters to tune
linreg_params = {
    'linreg__normalize': [True, False]
}

# Perform grid search cross-validation to find best hyperparameters
linreg_grid = GridSearchCV(linreg_pipe, linreg_params, cv=10, scoring='r2')
linreg_grid.fit(X_train_LR, y_train_LR)

# Train model with best hyperparameters
linreg_model = linreg_grid.best_estimator_
linreg_model.fit(X_train_LR, y_train_LR)

# Evaluate model performance
linreg_scores = cross_val_score(linreg_model, X_test_LR, y_test_LR, cv=10, scoring='r2')
print('Average R^2 score:', linreg_scores.mean() )

# Parameter choice 
print('Optimal Paraemter Choices :',linreg_model.get_params(deep=True))

#Model coefficents 
print('Model Coeffcients :',linreg_model.named_steps['linreg'].coef_)

# Plot feature coeff
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(linreg_model.named_steps['linreg'].coef_, marker='o', linestyle='')

ax.set_xticks(np.arange(8))
# Add feature names as tick labels
ax.set_xticklabels(['Number of Scenes','Lines in Script' ,'Directions by Director' , 'Number of Actors' , 'Duration of Episode' ,'PCA Directors 1', 'PCA Directors 2', 'PCA Actors 1'], rotation=90)

# Add labels and title
ax.set_xlabel("Features")
ax.set_ylabel("Coefficient")
ax.set_title("Linear Regression Feature Coefficients")
ax.legend()

plt.show()



################################################################################################
# Random Forest #
#################

# Define hyperparameters to tune
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, 9],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 3, 4]
}

# Perform grid search cross-validation to find best hyperparameters
rf_grid = GridSearchCV(RandomForestRegressor(), rf_params, cv=10, scoring='r2')
rf_grid.fit(X_train, y_train)

# Train model with best hyperparameters
rf_model = rf_grid.best_estimator_
rf_model.fit(X_train, y_train)

# Evaluate model performance
rf_scores = cross_val_score(rf_model, X_test, y_test, cv=10, scoring='r2')
print('Random Forest R^2 scores:', rf_scores)
print('Average R^2 score:', rf_scores.mean())

# Get feature importances from the trained model
importances = rf_model.feature_importances_

# Get a list of feature names
feature_names = X_train.columns

# Create a dataframe with feature names and their importances
feature_importances = pd.DataFrame({'feature': feature_names, 'importance': importances})

# Sort features by importance (descending order)
feature_importances = feature_importances.sort_values('importance', ascending=False)

# Print the top 10 features and their importances
print(feature_importances.head(10))

# Parameter choice 
print('Optimal Paraemter Choices :',rf_model.get_params(deep=True))


# Parameter choice 
print('Optimal Paraemter Choices :',rf_model.get_params(deep=True))


# Plot the R2 scores for each method
plt.plot(linreg_scores, label='Linear Regression')
plt.plot(rf_scores, label='Random Forest Regressor')

# Add a title, x and y labels, and a legend
plt.title("R2 scores across all cross folds")
plt.xlabel("Cross fold")
plt.ylabel("R2 score")
plt.legend()

# Show the plot
plt.show()




# Plot feature importances
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(feature_importances['feature'], feature_importances['importance'], color='b')

# Add labels and title
ax.set_xlabel("Features")
ax.set_ylabel("Importance")
ax.set_title("Random Forest Regressor Feature Importance")

plt.xticks(rotation=90)
plt.show()


# In[27]:



# Actor,writer,director Dataset #
#################################
# Split the data into training and testing sets
train_data = imdb_data_modeling_Int.sample(frac=0.7, random_state=32)
test_data = imdb_data_modeling_Int.drop(train_data.index)

# Define the independent variables and the target variable
X_train = train_data.drop(['imdb_rating','PCA_Setting_Predictors2','PCA_Setting_Predictors3'], axis=1)
X_test = test_data.drop(['imdb_rating','PCA_Setting_Predictors2','PCA_Setting_Predictors3'], axis=1)
y_train = train_data['imdb_rating']
y_test = test_data['imdb_rating']


################################################################################
# Linear Regression#
####################

# Feature Selection #
#####################

# Define LassoCV model with 5-fold cross-validation
lasso_model = LassoCV(cv=5)

# Fit the model on the training data
lasso_model.fit(X_train, y_train)

# Print selected features and their coefficients
selected_features = X_train.columns[lasso_model.coef_ != 0]
print("Selected features:", selected_features)
print("Coefficients:", lasso_model.coef_[lasso_model.coef_ != 0])


# Plot feature coefficients
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(lasso_model.coef_, marker='o', linestyle='')

# Add feature names as tick labels
ax.set_xticklabels(X_train.columns, rotation=90)

# Add labels and title
ax.set_xlabel("Features")
ax.set_ylabel("Coefficient")
ax.set_title("Lasso Feature Coefficients")

plt.show()

# Define the independent variables and the target variable
X_train_LR= train_data.loc[:,['director.Greg Daniels', 'director.Paul Feig', 'director.Tucker Gates',
       'writer.B.J. Novak', 'writer.Charlie Grandy', 'writer.Gene Stupnitsky',
       'writer.Greg Daniels', 'writer.Lee Eisenberg', 'writer.Mindy Kaling',
       'actor.Andy', 'actor.Angela', 'actor.Creed', 'actor.Darryl',
       'actor.Kelly', 'actor.Meredith', 'actor.Michael', 'actor.Oscar',
       'actor.Ryan', 'PCA_Setting_Predictors1']]
X_test_LR = test_data.loc[:,['director.Greg Daniels', 'director.Paul Feig', 'director.Tucker Gates',
       'writer.B.J. Novak', 'writer.Charlie Grandy', 'writer.Gene Stupnitsky',
       'writer.Greg Daniels', 'writer.Lee Eisenberg', 'writer.Mindy Kaling',
       'actor.Andy', 'actor.Angela', 'actor.Creed', 'actor.Darryl',
       'actor.Kelly', 'actor.Meredith', 'actor.Michael', 'actor.Oscar',
       'actor.Ryan', 'PCA_Setting_Predictors1']]
y_train_LR = train_data['imdb_rating']
y_test_LR = test_data['imdb_rating']



# Define pipeline
linreg_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('linreg', LinearRegression())
])

# Define hyperparameters to tune
linreg_params = {
    'linreg__normalize': [True, False]
}

# Perform grid search cross-validation to find best hyperparameters
linreg_grid = GridSearchCV(linreg_pipe, linreg_params, cv=10, scoring='r2')
linreg_grid.fit(X_train_LR, y_train_LR)

# Train model with best hyperparameters
linreg_model = linreg_grid.best_estimator_
linreg_model.fit(X_train_LR, y_train_LR)

# Evaluate model performance
linreg_scores = cross_val_score(linreg_model, X_test_LR, y_test_LR, cv=10, scoring='r2')
print('Average R^2 score:', linreg_scores.mean() )

# Parameter choice 
print('Optimal Paraemter Choices :',linreg_model.get_params(deep=True))

#Model coefficents 
print('Model Coeffcients :',linreg_model.named_steps['linreg'].coef_)



# Plot feature coeff
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(linreg_model.named_steps['linreg'].coef_, marker='o', linestyle='')

ax.set_xticks(np.arange(19))
# Add feature names as tick labels
ax.set_xticklabels(['Dir.Greg Daniels', 'Dir.Paul Feig', 'Dir.Tucker Gates',
       'Writer. B.J. Novak', 'Writer. Charlie Grandy', 'Writer. Gene Stupnitsky',
       'Writer. Greg Daniels', 'Writer. Lee Eisenberg', 'Writer. Mindy Kaling',
       'Actor. Andy', 'Actor. Angela', 'Actor. Creed', 'Actor. Darryl',
       'Actor. Kelly', 'Actor. Meredith', 'Actor. Michael','Actor. Oscar', 'Actor. Ryan',
       'PCA Setting Predictors 1'], rotation=90)

# Add labels and title
ax.set_xlabel("Features")
ax.set_ylabel("Coefficient")
ax.set_title("Linear Regression Feature Coefficients")
ax.legend()

plt.show()



################################################################################################
# Random Forest #
#################

# Define hyperparameters to tune
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, 9],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 3, 4]
}

# Perform grid search cross-validation to find best hyperparameters
rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=10, scoring='r2')
rf_grid.fit(X_train, y_train)

# Train model with best hyperparameters
rf_model = rf_grid.best_estimator_
rf_model.fit(X_train, y_train)

# Evaluate model performance
rf_scores = cross_val_score(rf_model, X_test, y_test, cv=10, scoring='r2')
print('Random Forest R^2 scores:', rf_scores)
print('Average R^2 score:', rf_scores.mean())

# Get feature importances from the trained model
importances = rf_model.feature_importances_

# Get a list of feature names
feature_names = X_train.columns

# Create a dataframe with feature names and their importances
feature_importances = pd.DataFrame({'feature': feature_names, 'importance': importances})

# Sort features by importance (descending order)
feature_importances = feature_importances.sort_values('importance', ascending=False)

# Print the top 10 features and their importances
print(feature_importances.head(10))

# Parameter choice 
print('Optimal Paraemter Choices :',rf_model.get_params(deep=True))



# Plot the R2 scores for each method
plt.plot(linreg_scores, label='Linear Regression')
plt.plot(rf_scores, label='Random Forest Regressor')

# Add a title, x and y labels, and a legend
plt.title("R2 scores across all cross folds")
plt.xlabel("Cross fold")
plt.ylabel("R2 score")
plt.legend()

# Show the plot
plt.show()



# Plot feature importance
fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(feature_importances.iloc[0:15,0], feature_importances.iloc[0:15,1], color='b')

# Add labels and title
ax.set_xlabel("Features")
ax.set_ylabel("Importance")
ax.set_title("Random Forest Regressor Feature Importance")

plt.xticks(rotation=90)
plt.show()


# In[43]:


# Reunion Episode Script Generation #
#####################################

################################################################################################
# Data Preparation#
###################

#Subset data written by Greg Daniels
Greg_Danials_Epsiodes=imdb_data_merged[imdb_data_merged['writer'].apply(lambda x: 'Greg Daniels' in x)]
Greg_transcript = pd.merge(Greg_Danials_Epsiodes, transcript, on=['season', 'episode'])


#Edit transcript so its in fromat of scene and speaker 
data = ["--Scene Start--"]
scene = 1

for index, row in Greg_transcript.iterrows():
    if scene != row['scene']:
        data.append("--Scene End--")
        data.append("")
        data.append("--Scene Start--")
        data.append(row['speaker'].strip() + ": " + row['line_text'].strip())
        scene += 1
    else:
        data.append(row['speaker'].strip() + ": " + row['line_text'].strip())

data.append("--Scene End--")
data.append("")
data.append("--Scene Start--")

# Saving all of the lines to a text file
with open('lines.txt', 'w') as filehandle:
    for listitem in data:
        filehandle.write('%s\n' % listitem)
        


# In[44]:



################################################################################################
# Modeling #
############

model_name = "124M"

# Download model
if not os.path.isdir(os.path.join("models", model_name)):
    print(f"Downloading {model_name} model...")
    gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/


# In[45]:


office_sess = gpt2.start_tf_sess()
gpt2.finetune(office_sess,
              'lines.txt',
              model_name=model_name,
              steps=1000,
              print_every=100,
              sample_every=100,
              save_every=500)   # steps is max number of training steps


# In[17]:


################################################################################################
# Generate Script Based on Prompt #
###################################
gpt2.generate(office_sess, length=250, temperature=0.8, prefix='Michael: This inflation stuff is geting out of hand, WE ARE GOING TO FILE FOR BANKRUPTCY')

