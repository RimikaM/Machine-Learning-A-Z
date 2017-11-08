# Random Forest Regression
# Step 1: Pick random K data points from the Training set
# Step 2: Build the decision tree associated to these K data points
# Step 3: Choose the number of Ntrees of trees you want to build and repeat the above steps
# Step 4: For a new data point, make each one of your Ntree trees predict the 
# value of Y to the data point in question and assign the new data point the average
# across all of the predicted values

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting the Random Forest Regression Model to the dataset
# Create your regressor here
from sklearn.ensemble import RandomForestRegressor
# n_estimators = num of trees in forest
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X, y)

# Predicting a new result
# prediction of ~160K, which is the correct salary
y_pred = regressor.predict(6.5)

# non continuous regression model
# Visualising the Random Forest Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.savefig("Random_Forest_py.png")