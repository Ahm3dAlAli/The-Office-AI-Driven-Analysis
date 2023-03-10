# 📺 TV Show Ratings Analysis 🎬

Welcome to my TV show ratings analysis project! This project aims to analyze various factors that affect the ratings of The Office TV show and provide recommendations to NBC Universal to increase their ratings.

## 📋 Table of Contents

<ul>
  <li><a href="#introduction">Introduction</a></li>
  <li><a href="#literature">Literature Review</a></li>
  <li><a href="#data">Data Collection and Preparation</a></li>
  <li><a href="#metho">Methodology</a></li>
  <li><a href="#analysis">Data Analysis</a></li>
  <li><a href="#recommendations">Recommendations</a></li>
  <li><a href="#script-generation">Script Generation</a></li>
</ul>
<a name="introduction"></a>

## 🎉 Introduction

TV show ratings play a critical role in determining the success of a show. The higher the ratings, the more successful the show. Thus, it is essential to understand the various factors that contribute to ratings and how they can be improved.

<a name="literature"></a>

## 📚 Literature Review

As part of this project, a literature review was conducted to find relevant articles related to the topic. Here are some of the most relevant ones:

Hunt et al. (2020) - The effect of diversity on television show ratings.
Rosenberg and VanMeter (2019) - The impact of runtime on television show ratings.
Bruns and Highfield (2017) - The influence of directors and writers on television ratings.
Zhang et al. (2016) - A study of the influence of directors and writers on television show ratings.
Carrell and Jerit (2012) - The impact of character development on television show ratings.
Wu et al. (2017) - Predicting TV show ratings using a support vector regression model.
Fagnan et al. (2019) - Predicting television show ratings using gradient boosting regression.
Lee and Kim (2016) - Predicting television show ratings using multiple linear regression.
Yu et al. (2017) - Predicting TV show ratings using sentiment analysis.
Seo et al. (2018) - Predicting television show ratings using social media sentiment analysis and machine learning techniques.

<a name="data"></a>

## 📊 Data Collection and Preparation

Data was collected from various sources, including IMDb, Kaggle, and Rotten Tomatoes. The data was then preprocessed and cleaned to remove missing values, outliers, and irrelevant features.

<a name="analysis"></a>


## 📝 Methodology

To investigate the factors that influence TV show ratings, we used a three-scheme approach that involved data preprocessing, model building, and analysis. The data were collected from a publicly available dataset containing information on TV shows including their ratings, cast, crew, and other relevant features. We preprocessed the data to handle missing values, outliers, and other data quality issues.

For scheme 1, we built a multiple linear regression model to explore the relationship between the predictors and the ratings. We used stepwise feature selection to identify the most significant variables and evaluate the model performance using R-squared (R2).

For scheme 2, we used the Random Forest Regressor to identify the optimal settings that lead to higher ratings. We also used feature importance ranking to identify the most influential variables that impact the ratings.

For scheme 3, we used the LASSO Linear Regression and Random Forest Regressor models to investigate the impact of directors, writers, and actors on TV show ratings. We evaluated the model performance using R2 and feature importance rankings.

To generate a script using GPT-2, we extracted a subset of data written by Greg Daniels and merged it with transcript data to create a new dataset. We reformatted the transcript data to be in the format of scenes and speakers. We then used the GPT-2 model to generate a script based on the optimal settings determined in scheme 2. We fine-tuned the model using the training data and saved the model at different steps to generate the sample script.
<a href="#metho"></a>

## 📈 Data Analysis

Various data analysis techniques were used to analyze the data and identify factors that influence TV show ratings. The analysis included modeling the influence of predictors on the ratings, determining the optimal settings that lead to higher ratings, and determining the optimal directors, writers, and actors for a higher rating.

<a name="recommendations"></a>

## 💬 Recommendations

Based on the analysis, recommendations were made to NBC Universal to increase their ratings. The recommendations include increasing the number of scenes, selecting the right year for airing the episode, keeping the number of actors to about 10 actors, considering the input of the director to ensure a positive sentiment score for the transcript, keeping the duration of the episode within a suitable range, considering the writer Greg Daniels and the director Paul Feig, including the actors Kelly, Daryl, Creed, Ryan, Oscar, Tobby, Phyllis, and Stanley, and considering Michael, Jim, and Dwight as the main characters.

<a name="script-generation"></a>

## 🎭 Script Generation

Finally, a GPT-2 model was utilized to generate a script based on the optimal settings. The
