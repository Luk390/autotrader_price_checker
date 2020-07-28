# Project Overview

* Created a linear regression model (Lasso) which can estimate the price of a petrol car up to 3 years old with an R2 score of 92.7 %
* Collected the data from AutoTrader using a webscraper
* Built a basic API using Flask

![alt text](https://github.com/Luk390/autotrader_price_checker/blob/master/images/senna.jpg "A one off McLaren commission")

## Code and Resources Used

* Python 3.7
* Packages: Pandas, Numpy, Scikit-Learn, Matplotlib, flask, json, pickle, missingno
* Flask API: https://towardsdatascience.com/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2
* Autotrader Scraper: https://github.com/liudvikasakelis/autotrader-scraper

## Data Cleansing

* The data was captured by scraping a search from autotrader. Five searches were done in total for petrol cars that were made and 2018 and later in Bristol, Birmingham, Manchester, Leeds and London.
* The data was fairly well complete from the scraper but needed basic amendments. Some of the categorical features needed coding to a more meaningful names, some exceptionally rare cars were dropped and "trim" feature was dropped due over 60% missing values.

## EDA

* A basic EDA was carried out, which revealed that a polynomial expansion was probably necessary and there was a significant issue with multicollinearity, so there was almost certainly going to be a need for some regularization.

![alt text](https://github.com/Luk390/autotrader_price_checker/blob/master/images/OLS.PNG "OLS Output")

## Models

A pipeline was created which preprocessed the data and then trained a Lasso model. The preprocessing consisted of using Scikit-learns's StandardScaler() on the numeric features and OneHotEncoder() on the categorical data. 

The lasso model achieved an R2 score of 92.7%.

## Productionisation

A flask API endpoint was then created and hosted on a local server. This takes in a request with a dataframe of features and then returns an estimated value.
