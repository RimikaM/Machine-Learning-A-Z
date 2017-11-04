# Polynomial Regression
# not a linear regression model

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
# find the correlations between the level and salary
# to find out if the employee is bluffing about the salary
# position is already encoded in level column
# so it doesn't need to be included in the matrix
# of features X
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
# doesn't make sense to split the data into 
# training and tests sets because there are only
# 10 observations
'''from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)'''

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting linear regression to the dataset
# this will be a reference to then be able to 
# compare the results of polynomial regression
# to the results of the linear regression reference
# space
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

# Fitting Polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
# used to make a new matrix of features X_poly
# poly_reg = PolynomialFeatures(degree = 2)
# poly_reg = PolynomialFeatures(degree = 3)
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
# include this fit into a multiple linear regression model
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the linear regression results
# actual data
plt.scatter(X, y, color = 'red')
# now plot prediction data
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.savefig('Linear_py.png')

# Visualising the polynomial regression degree 2 results
# actual data
plt.scatter(X, y, color = 'red')
# now plot prediction data
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Degree 2 Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.savefig('Polynomial_2_py.png')

# Visualising the polynomial regression degree 3 results
# actual data
plt.scatter(X, y, color = 'red')
# now plot prediction data
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Degree 3 Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.savefig('Polynomial_3_py.png')

# Visualising the polynomial regression degree 4 results
# perfect model!
# actual data
plt.scatter(X, y, color = 'red')
# now plot prediction data
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Degree 4 Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.savefig('Polynomial_4_py.png')

# Predicting a new result with Linear Regression
# predict one salary of level 6.5
lin_reg.predict(6.5)

# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))