# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
# data of columns except last
X = dataset.iloc[:,:-1].values
# data of last column
y = dataset.iloc[:,3].values


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
# split arrays into random train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

# Feature scaling
''' from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
# fit is not needed because it is already fitted to the training set
X_test = sc_X.transform(X_test)'''