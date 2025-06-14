# Mobile Phone Price Prediction using Logistic Regression
# 0 = Low, 1 = Medium, 2 = High, 3 = Very High

import pandas as pd

dataset = pd.read_csv('dataset.csv')
# print(dataset.head())
# print(dataset.isnull().sum()) # No empty values found

X = dataset.iloc[:,:-1]
Y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train , X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)

# print(X_train.head())

from sklearn.linear_model import LogisticRegression
reg = LogisticRegression(max_iter=10000)
reg.fit(X_train,Y_train)
# Tried the regularizatioon and no need of regularization

Y_pred = reg.predict(X_test)

from sklearn.metrics import accuracy_score,precision_score,recall_score
print('Accuracy:', accuracy_score(Y_test,Y_pred))
print('Precision:', precision_score(Y_test,Y_pred,average='weighted'))
print('Recall:', recall_score(Y_test,Y_pred,average='weighted'))

import pickle
pickle.dump(reg,open('reg.pkl','wb'))