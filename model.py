# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 08:10:52 2020

@author: lukeb
"""


import pickle
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer, make_column_transformer

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

Now we have an idea of which model to use and the preprocessing steps that need
to be conducted. The next step will involve creating a pipeline of preprocessing
and a gridsearch for parameter selection.

"""

preprocess = make_column_transformer(
    (StandardScaler(), ['co2Emissions', 'mileage', 'engine_size']),
    (PolynomialFeatures(), ['co2Emissions', 'mileage', 'engine_size']),
    (OneHotEncoder(sparse=False, handle_unknown='ignore'), ['body_type', 
     'condition', 'doors', 'isTradeSeller', 'make', 'manufactured_year', 
     'model', 'seats', 'transmission', 'location'])
    )


param_grid = {
    'lasso__alpha': [0.001, 0.01, 0.1, 1, 10, 100],
    'columntransformer__polynomialfeatures__degree': [2,3]
    }
    
    
pipe2 = make_pipeline(
        preprocess,
        Lasso(max_iter=10000)
        )

grid = GridSearchCV(pipe2, param_grid=param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)
print(grid.score(X_train, y_train))
print(grid.score(X_test, y_test))
print(grid.best_params_)
#print(grid.mean_absolute_error)

"""
The model only produced an MAE of 92.7% not much of an improvement on the above
but the gridsearch provides more confidence in how reliable the model will be
on unseen data.

"""

pickl = {'model': grid.best_estimator_}
pickle.dump(pickl, open('models/model_file' + '.p', 'wb'))


file_name = "models/model_file.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']
    
model.predict(X_test.iloc[0:1,:])



"""


"""
