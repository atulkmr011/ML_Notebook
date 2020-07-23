# This is an example of Multiple Linear Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# dataset import
dataset = pd.read_csv('Startups.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 4]


#Convert the column into categorical columns

states=pd.get_dummies(X['State'],drop_first=True) #here states is the variable required and state is the name of column

# Drop the state coulmn because we have hot endoded that
X=X.drop('State',axis=1)

# Now concatinating the dummy variables
X=pd.concat([X,states],axis=1)


# Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# checking the r2 score the best the close to 1
from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)