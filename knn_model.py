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

from sklearn.neighbors import KNeighborsClassifier as KNN

classifier = KNN()
classifier.fit(X_train,y_train)

#KNeighborsClassifier()

print(classifier,X_train,y_train,X_test,y_test,train=True)