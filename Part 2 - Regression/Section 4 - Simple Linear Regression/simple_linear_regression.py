# Simple Linear Regression
# simplest machine learning model
# straight line prediction

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
# 30 observations, 20 in training, 10 in test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling - library takes care of it for us
# for simple linear regression
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# The machine is 'regressor', which will be able to 
# predict the salary based on the years of experience

# Fitting Simple Linear Regression in the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
# fit regressor to training sets
# the machine is learning the correlation
# of experience with salary using the fit function
regressor.fit(X_train, y_train)

# Predicting the Test set results
# create a vector that will contain the predictions
# of the test set salaries and we will put all these 
# predicted salaries into a single vector y_pred
#
# y_pred is the vector of predictions of the 
# dependent variable
#
# Predict method will make the predictions of the 
# salaries of some observations, so you need to 
# specify which observations you want to make
# the predictions
#
# y_test contains real salaries and
# y_pred contains predicted salaries
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
# compare real salaries to predicted salaries based
# on the observations of the training set
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.savefig('Salary_Experience_Train_Graph_py.png')

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
# compare real salaries to predicted salaries based
# on the observations of the test set
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.savefig('Salary_Experience_Test_Graph_py.png')