import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# split dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


# scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# K Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train) #model fitting

predict = knn.predict(X_test)
print(predict[:5]) #first 5 predictions

#probability of a prediction, 0 to 1 for each class
y_pred_prob = knn.predict_proba(X_test)
print(y_pred_prob[:5]) #probability of first 5 predictions

#accuracy, proportion of data points that match the prediction
print(knn.score(X_test, y_test))

#print((predict==y_test.values).sum()/y_test.size)

# #confusion matrix, berapa nilai true yg benar terprediksi
# matrix = confusion_matrix(y_test, predict, labels=['iris-setosa', 'iris-versicolor', 'iris_virginica'])
# print(matrix)

# plot_confusion_matrix(knn, X_test, y_test, cmap=plt.cm.Blues)
# plt.show()