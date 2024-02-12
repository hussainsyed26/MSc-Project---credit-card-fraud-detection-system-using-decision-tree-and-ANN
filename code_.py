#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import necessary libraries
import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import init_notebook_mode, iplot
import PIL
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Initialize Plotly for offline plotting
init_notebook_mode(connected=True)

# Read the dataset
data_df = pd.read_csv('card_transdata.csv')

# Display first few rows of the dataset
data_df.head()

# Get summary statistics of the dataset
data_df.describe()

# Check for missing values
total = data_df.isnull().sum().sort_values(ascending=False)
percent = (data_df.isnull().sum() / data_df.isnull().count() * 100).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data

# Display information about the dataset
data_df.info()

# Check for missing values again
data_df.isnull().sum()

# Get the shape of the dataset
data_df.shape

# Visualize the distribution of the target variable 'fraud'
sns.countplot(data_df['fraud'])

# Split the dataset into features (x) and target variable (y)
x = data_df.drop("fraud", axis=1)
y = data_df["fraud"]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Instantiate Decision Tree Classifier
dtc = DecisionTreeClassifier()

# Fit the model on the training data
dtc.fit(x_train, y_train)

# Predict the target variable on the testing data
y_pred = dtc.predict(x_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_pred, y_test)

# Display accuracy
print("Accuracy:", accuracy)

# Create a confusion matrix
cm = confusion_matrix(y_pred, y_test)

# Plot the confusion matrix
sns.heatmap(cm, annot=True)

# Feature scaling for neural network
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Encode the target variable for neural network
labelencoder_y = LabelEncoder()
y_train = labelencoder_y.fit_transform(y_train)

# Build Artificial Neural Network (ANN)
ANN = Sequential()
ANN.add(Dense(1, activation='sigmoid'))
ANN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the ANN
hist = ANN.fit(x_train, y_train, epochs=10, batch_size=1, verbose=1)

# Predict the target variable on the testing data using ANN
y_pred = ANN.predict(x_test)
y_pred = (y_pred > 0.5)

# Create confusion matrix for ANN
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
sns.heatmap(cm, annot=True)

