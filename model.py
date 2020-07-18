# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 08:10:52 2020

@author: lukeb
"""


import pickle
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

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

ridge = Ridge().fit(X_train_scaled_encoded, y_train)
print(f"Training Score: {ridge.score(X_train_scaled_encoded, y_train)}")
print(f"Test Score: {ridge.score(X_test_scaled_encoded, y_test)}")

lasso = Lasso().fit(X_train_scaled_encoded, y_train)
print(f"Training Score: {lasso.score(X_train_scaled_encoded, y_train)}")
print(f"Test Score: {lasso.score(X_test_scaled_encoded, y_test)}")

"""
Without any parameter tuning or polynomial expansions, we are already getting 
and R2 of 92% for both models.

Looking at the scatter graphs it was obvious that there are some polynomial
relationships between the independent variables and the target variable. So a
basic polynomial expansion will implemented.
"""


poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(X_train_scaled_encoded)
x_test_poly = poly.transform(X_test_scaled_encoded)
poly_ridge = Ridge().fit(x_poly, y_train)
print(f"Train score {poly_ridge.score(x_poly, y_train)}")
print(f"Test score {poly_ridge.score(poly.transform(X_test_scaled_encoded), y_test)}")

poly_lasso = Lasso().fit(x_poly, y_train)
print(f"Train score {poly_lasso.score(x_poly, y_train)}")
print(f"Test score {poly_lasso.score(poly.transform(X_test_scaled_encoded), y_test)}")
print(f"Number of features used: {np.sum(poly_lasso.coef_ != 0)}")

"""
This pushed the score up slightly to approx. 93% for both models. Surprisingly
there isn't much evidence of overfitting despite the very high number of features.

The next step will involve a gridsearch cross-validation to push the accuracy
up as high as possible. Due to the high number of features the Lasso model will
be chosen to try to slim down the high number of collinear variables.
"""

param_grid = {'lasso__alpha':[0.001, 0.01, 0.1, 1, 10, 100]}
pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso', Lasso(max_iter=10000))
        ])


grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(x_poly, y_train)
print(f"Best Score: {grid.best_score_}")
print(f"Test set score: {grid.score(x_test_poly, y_test)}")
print(f"Best Parameters: {grid.best_params_}")


"""

I think I need to create a new gridsearch. 
1 - create x_poly with only x_train_numeric_features
2 - encode that data
3 - Scale the data - potentially between values 1 and 0???
4 - train model
5 - use grid.score that will do the same transformations on the test set.

ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 2791966638.564788, tolerance: 362203124.1697053
  positive)
C:\Users\lukeb\Anaconda3\lib\site-packages\sklearn\linear_model\coordinate_descent.py:475: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 713460439.3617058, tolerance: 371229675.8593116
  positive)
C:\Users\lukeb\Anaconda3\lib\site-packages\sklearn\linear_model\coordinate_descent.py:475: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 623489921.3752999, tolerance: 329055837.96231246
  positive)
C:\Users\lukeb\Anaconda3\lib\site-packages\sklearn\linear_model\coordinate_descent.py:475: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 2524248099.4247904, tolerance: 344050772.2479679
  positive)
C:\Users\lukeb\Anaconda3\lib\site-packages\sklearn\linear_model\coordinate_descent.py:475: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 2699671243.9496603, tolerance: 356663968.8917436
  positive)
Best Score: 0.950836582824999
Test set score: 0.9399934094581518
Best Parameters: {'lasso__alpha': 100}

"""







