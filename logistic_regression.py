import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

# import data
dataset = pd.read_csv('mushrooms.csv')

# sneak peak data
print(dataset.describe())
print(dataset.info())
print(dataset.head())

# checking for any null
print(dataset.isnull().sum())

# separating class as response, and others column as predictor
X = dataset.drop('class',axis=1) # Predictors
Y = dataset['class'] #Response
print(X.head())

# encode categorical data into number
from sklearn.preprocessing import LabelEncoder
Encoder_X = LabelEncoder() 
for col in X.columns:
    X[col] = Encoder_X.fit_transform(X[col])
Encoder_Y=LabelEncoder()
Y = Encoder_Y.fit_transform(Y)
print(X.head())

# # separating train and target values
# target = dataset['class']
# train = dataset[['gill-size', 'gill-color']]
# print(train.shape)
# print(target.shape)


# # In logistic regression, the link function is the sigmoid. We can implement this really easily.
# # The sigmoid function has special properties that can result values in the range [0,1]. 
# # So you have large positive values of X, the sigmoid should be close to 1, 
# # while for large negative values, the sigmoid should be close to 0.

# def sigmoid(theta, X):
#     X = np.array(X)
#     theta = np.asarray(theta)
#     return((1/(1+math.e**(-X.dot(theta)))))


# # Function for the cost function of the logistic regression.
# def cost(theta, X, Y):
#     first = np.multiply(-Y, np.log(sigmoid(theta,X)))
#     second = np.multiply((1 - Y), np.log(1 - sigmoid(theta,X)))
#     return np.sum(first - second) / (len(X))   

# # calculates gradient of the log-likelihood function
# def log_gradient(theta, X, Y):
#     first_calc = sigmoid(theta, X) - np.squeeze(Y).T
#     final_calc = first_calc.T.doc(X)
#     return(final_calc.T)

# # function performing gradient descent
# def gradient_Descent(theta, X, Y, itr_val, learning_rate=0.00001):
#     cost_iter = []
#     cost_val=cost(theta,X,Y)
#     cost_iter.append([0,cost_val])
#     itr = 0
#     while(itr < itr_val):
#         theta = theta - (0.01 * log_gradient(theta, X, Y))
#         cost_val = cost(theta, X, Y)
#         cost_iter.append([i, cost])
#         itr += 1
#     return theta

# def pred_values(theta, X, hard=True):
#     X = (X - np.mean(X, axis=0))/np.std(X, axis=0)
#     pred_prob = sigmoid(theta, X)
#     pred_value = np.where(pred_prob >= .5, 1, 0)
#     return pred_value

# theta = np.zeros
# theta = np.asmatrix(theta)
# theta = theta.T
# target = np.asmatrix(target).T
# y_test = list(target)

# params = [10, 20, 30, 50, 100]
# for i in range(len(params)):
#     th = gradient_Descent(theta,train,target,params[i])
#     y_pred = list(pred_values(th, train))
#     score = float(sum( 1 for x, y in zip(y_pred, y_test) if x == y)) / len(y_pred)
#     print("The accuracy after " + '{}'.format(params[i]) + " iteration is " + '{}'.format(score))

# from sklearn.linear_model import LogisticRegression
# clf = LogisticRegression()
# print(clf.fit(train, target))
# print(clf.score(train, target))


# split dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


# Logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
lr=LogisticRegression(max_iter=10000)
lr.fit(X_train,y_train)
pred_1=lr.predict(X_test)
score_1=accuracy_score(y_test,pred_1)
print(pred_1)
print(score_1)