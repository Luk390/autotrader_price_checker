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
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures

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

ridge = Ridge().fit(X_train_scaled_encoded, y_train)
print(f"Training Score: {ridge.score(X_train_scaled_encoded, y_train)}")
print(f"Test Score: {ridge.score(X_test_scaled_encoded, y_test)}")

lasso = Lasso().fit(X_train_scaled_encoded, y_train)
print(f"Training Score: {lasso.score(X_train_scaled_encoded, y_train)}")
print(f"Test Score: {lasso.score(X_test_scaled_encoded, y_test)}")

"""
Without any parameter tuning or polynomial expansions, we are already getting 
and R2 of 95% and 94%.
"""
#do basic with polynomial 

"""
The below works but, there are so many features. I think make a pipeline and poly
the numeric features, then add in categorical. See if that ups the score

Ridge poly 
Train score 0.9965684846094797
Test score 0.9425541510033445
Lasso poly
Train score 0.9961384183740127
Test score 0.9448654566082518

Also Lasso didn't converge

Pipeline!!! exciting!
"""

poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(X_train_scaled_encoded)

poly_ridge = Ridge().fit(x_poly, y_train)
print(f"Train score {poly_ridge.score(x_poly, y_train)}")
print(f"Test score {poly_ridge.score(poly.transform(X_test_scaled_encoded), y_test)}")

poly_lasso = Lasso().fit(x_poly, y_train)
print(f"Train score {poly_lasso.score(x_poly, y_train)}")
print(f"Test score {poly_lasso.score(poly.transform(X_test_scaled_encoded), y_test)}")

#when best model selected - create a pipeline which scales,encodes, cross-validates and gridseaches
#Pipeline needed so you don't leak information during the cross validation



















