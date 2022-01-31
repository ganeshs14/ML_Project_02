import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import re

import warnings
warnings.filterwarnings('ignore')

# Training Dataset
df_train = pd.read_csv('Training Data.csv')

df_train.drop('Loan_ID', axis=1, inplace=True)

df_train.Gender.fillna(df_train.Gender.mode()[0],inplace=True)
df_train.Gender.replace(['Male','Female'], [0,1],inplace=True) 

df_train.Married.fillna('Yes',inplace=True)
df_train.Married.replace(['Yes','No'], [1, 0], inplace=True)

df_train.Dependents.fillna('0', inplace=True)
df_train.Dependents.replace(['0', '1', '2', '3+'], [0, 1, 2, 3], inplace=True)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df_train.Education = le.fit_transform(df_train.Education)

df_train.Self_Employed.fillna(df_train.Self_Employed.mode()[0], inplace=True)
df_train.Self_Employed = le.fit_transform(df_train.Self_Employed)

df_train.LoanAmount.fillna(df_train.LoanAmount.median(), inplace=True)

df_train.Loan_Amount_Term.fillna(df_train.Loan_Amount_Term.median(), inplace=True)

df_train.Credit_History.fillna(df_train.Credit_History.median(), inplace=True)

df_train.Property_Area = le.fit_transform(df_train.Property_Area)

df_train.Loan_Status = le.fit_transform(df_train.Loan_Status)



from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import train_test_split, RandomizedSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X = df_train.drop('Loan_Status',axis=1) 

y = df_train.Loan_Status

etc = ExtraTreesClassifier()
etc.fit(X, y)
imp_features = pd.Series(etc.feature_importances_,index=X.columns)
new_X = df_train[list(imp_features.nlargest(10).index)]

X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=0.30, random_state=42)

rfc_model = RandomForestClassifier()
rfc_model.fit(X_train, y_train)

rfc_y_pred = rfc_model.predict(X_test)

rfc_accuracy = (100 *accuracy_score(y_test, rfc_y_pred)).round(2)
print(f"Accuracy of Random Forest Classifier model : {rfc_accuracy} %\n")
print('Confusion Matrix : \n', confusion_matrix(y_test, rfc_y_pred),'\n')
print("Classification Report : \n",classification_report(y_test, rfc_y_pred))


n_estimators = [int(i) for i in range(100,1001,100)]
max_features = ['auto', 'sqrt']
max_depth = [i for i in np.linspace(start=5, stop=30, num=6)]
min_samples_split = [2, 5, 7, 10, 20]
min_samples_leaf = [1, 2, 3, 5, 10]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf
              }

random_s_cv = RandomizedSearchCV(estimator=rfc_model,
                               param_distributions=random_grid,
                               scoring = 'neg_mean_squared_error',
                               n_jobs = 1,
                               random_state = 0,
                               verbose=2,
                               cv=5)

random_s_cv.fit(X_train, y_train)

y_pred = random_s_cv.predict(X_test)

rf_accuracy = (100 *accuracy_score(y_test, y_pred)).round(2)
print(f"\n\nAccuracy of Random Forest Classifier Model after Hyper-parameter tuning : {rf_accuracy} %\n")
print('Confusion Matrix : \n', confusion_matrix(y_test, y_pred),'\n')
print("Classification Report : \n",classification_report(y_test, y_pred))

import pickle

pickle.dump(random_s_cv, open('RFC_model.pkl', 'wb'))