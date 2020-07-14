# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 08:10:52 2020

@author: lukeb
"""

"""
* Do I need the column names?? - If not, then I can just do the encoding/scaling after a little EDA.

Collinearity is an issue.
    * Use either Recursive Feature Selection
    * Or other options in the book

For myself

* Work out how to properly do onehotencoding. Is it possible to fall into the trap zedstatistics mentioned?

* How do you unstandardise the coefficients?

* which method to measure accuracy of model MAE?

* Things to do
    * Use stats models to do a simple linear regression and explain the output - # Consider going back to do this
    * Check for variables with coefficients close to 0 and p value of the t-statistic greater than 0.05 - This indicates the interpretability of the coefficients
    * Create pipeline - Need experience using it
    * Smartly select features. Explain why the features have been selected

"""

import pickle

with open('train.pickle', 'rb') as f:
    X_train, y_train = pickle.load(f)

with open('test.pickle', 'rb') as f:
    X_test, y_test = pickle.load(f)
    
with open('train_scaled.pickle', 'rb') as f:
    X_train_scaled_encoded = pickle.load(f)
    
with open('test_scaled.pickle', 'rb') as f:
    X_test_scaled_encoded = pickle.load(f)
    
"""
We will start with doing some basic models to get an idea of how effective
regularization is on this dataset. If this doesn't produce the desired results
more sophisticated feature selection will be carried out, such as recursive
feature selection (computationally expensive).

"""
#import scaled and encoded data from jupyter - Done

#do basic models to get a baseline score using L1 and L2 regularization

#do basic with polynomial 

#when best model selected - create a pipeline which scales,encodes, cross-validates and gridseaches
#Pipeline needed so you don't leak information during the cross validation



















