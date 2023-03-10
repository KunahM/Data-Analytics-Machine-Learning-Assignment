#!/usr/bin/env python
# coding: utf-8

# 
# # EE514 assignment part 1 : Initial Setup and Starter Code

# In[1612]:


#import library 

import pandas as pd
import numpy as np
from pathlib import Path
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import statistics
import csv
import os

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report


# In[1613]:


# DataSet file path location :

data_dir = 'C:/Users/Kunah/OneDrive/Documents/Dublin City University/Assignment/DA && ML/sensor_starter/data'
os.chdir(data_dir)


# In[1614]:


# Utility functions :

def load_data_for_user(uuid):
    return pd.read_csv(data_dir + '/' + (uuid + '.features_labels.csv'))

def get_features_and_target(df, feature_names, target_name):
    
    # select out features and target columns and convert to numpy
    X = df[feature_names].to_numpy()
    y = df[target_name].to_numpy()
    
    # remove examples with no label
    has_label = ~np.isnan(y)
    X = X[has_label,:]
    y = y[has_label]
    return X, y


# In[1615]:


# Data for one user :

df = load_data_for_user('99B204C0-DD5C-4BB7-83E8-A37281B8D769')
df.head()


# In[1616]:


# Columns available in the given DataSet :

print(df.columns.to_list())


# In[1675]:


# Features can be selected accordingly by selecting labels 
# FIX_Walking, #Bicycling , #Sitting
# select targets
# selecting columns for walking 

acc_sensors = [p for p in df.columns if 
               p.startswith('raw_acc:') or 
               p.startswith('watch_acceleration:')]

target_column = 'label:FIX_walking'
print(f'target_column')
# Raw data is used which is reported by the user since there are no columns for Bicycling and Sitting. 


# In[1676]:


# Training data model :

X_train, y_train = get_features_and_target(df, acc_sensors, target_column)
print(f'{y_train.shape[0]} examples with {y_train.sum()} positives')


# In[1677]:


#SimpleImputer replaces all the Nan's with mean value of the column
#StandardScaler scales the data to have value between 0 and 1 and make the data normally distributed.

scaler = StandardScaler()
imputer = SimpleImputer(strategy='mean')

X_train = scaler.fit_transform(X_train)
X_train = imputer.fit_transform(X_train)


# # Fitting a model - Training Accuracy
# Let's fit a logistic regression model to this user. We can then test it's predictive power on a different user
# Let's see the accuracy on the training set. The score function can be used to do this:

# In[1678]:


# training model :

kun = LogisticRegression(solver='liblinear', max_iter=1000, C=0.5)
kun.fit(X_train, y_train)


# In[1679]:


print(f'Training accuracy of trained model: {kun.score(X_train, y_train):0.4f}')

1 - y_train.sum() / y_train.shape[0]


# # Balanced Accuracy

# In[1680]:


# Balanced Accuracy calculation:

y_user1_prediction = kun.predict(X_train)
print(f'Balanced accuracy of train: {metrics.balanced_accuracy_score(y_train, y_user1_prediction):0.4f}')


# ## Testing the model
# 
# Ok, it seems our model has fit the training data well. How well does it perform on unseen test data? Let's load the data in for a different user.

# In[1540]:


#load dataframe 

df_test = load_data_for_user('CA820D43-E5E2-42EF-9798-BE56F776370B')
X_test, y_test = get_features_and_target(df_test, acc_sensors, target_column)
print(f'{y_train.shape[0]} examples with {y_train.sum()} positives')


# We also need to preprocess as before. **Note**: we are using the scaler and imputer fit to the training data here. It's very important that you do not call `fit` or `fit_transform` here! Think about why.

# In[1541]:


X_test = imputer.transform(scaler.transform(X_test))


# ## Test accuracy

# In[1542]:


print(f'Test accuracy: {kun.score(X_test, y_test):0.4f}')


# # __Section 2: Improving the Test Set__ 

# In[1543]:


# Selecting 5 different users and testing the model on each indiviually
# Evaluating mean and variance of the balanced accuracy

u_str =  ['7CE37510-56D0-4120-A1CF-0E23351428D2', '27E04243-B138-4F40-A164-F40B60165CF3', 'BEF6C611-50DA-4971-A040-87FB979F3FC1','A5CDF89D-02A2-4EC1-89F8-F534FDABDD96','1DBB0F6F-1F81-4A50-9DF4-CD62ACFA4842']
x_list = []



def store_list(x) :          # function to store list
    x_list.append(x)
    
    
    
for u in u_str:
    df_test = load_data_for_user(u)
    X_test, y_test = get_features_and_target(df_test, acc_sensors, target_column)
    print("UUID : ", u)
    print(f'{y_train.shape[0]} examples with {y_train.sum()} positives')
    X_test = imputer.transform(scaler.transform(X_test))
    print(f'Test accuracy: {kun.score(X_test, y_test):0.4f}')
    y_pred = kun.predict(X_test)
    print(f'Balanced accuracy (train): {metrics.balanced_accuracy_score(y_test, y_pred):0.4f}')
    x = float(format(metrics.balanced_accuracy_score(y_test, y_pred)))
    store_list(x)
    print(" ")

x_sum = sum(x_list)
x_mean = x_sum/len(x_list)
print("Mean of Balanced Accuracy : ",x_mean)

var = statistics.variance(x_list)
print("Variance of Balanced Accuracy : ",var)


# In[1544]:


kun = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)
kun.fit(X_train, y_train)

print(f'SVM Training accuracy for C = 1.0 : {kun.score(X_train, y_train):0.4f}')

y_user1_prediction = kun.predict(X_train)
print(f'Balanced accuracy of train: {metrics.balanced_accuracy_score(y_train, y_user1_prediction):0.4f}')


# # __Section 4: Increase training data__

# In[1545]:


#Combining the 5 users and putting in data frame df_combined

u_str = ['7CE37510-56D0-4120-A1CF-0E23351428D2', '27E04243-B138-4F40-A164-F40B60165CF3', 'BEF6C611-50DA-4971-A040-87FB979F3FC1','A5CDF89D-02A2-4EC1-89F8-F534FDABDD96','1DBB0F6F-1F81-4A50-9DF4-CD62ACFA4842']


combined_csv = pd.concat([pd.read_csv(data_dir + '/' + (u + '.features_labels.csv')) for u in u_str])
df_csv = pd.DataFrame(combined_csv)  
df_csv.to_csv("Kunah_csv.csv", index=False, encoding='utf-8-sig')


# In[1546]:


df_combined = pd.read_csv(data_dir +'/'+'Kunah_csv.csv')


# In[1563]:


from sklearn.model_selection import train_test_split


# In[1548]:


X_test, y_test = get_features_and_target(df_combined, acc_sensors, target_column)


# In[1549]:


# Splitting the data set into 80-20 (80 = Training set ,20 = Validation set ) INTRODUCING VALIDATION SET
# 0.25 x 0.8 = 0.2

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)


# In[1550]:


scaler = StandardScaler()
imputer = SimpleImputer(strategy='mean')

X_train = scaler.fit_transform(X_train)
X_train = imputer.fit_transform(X_train)


# In[1551]:


kun = LogisticRegression(solver='liblinear', max_iter=1000, C=1.0)
kun.fit(X_train, y_train)


# In[1552]:


print(f'LR with C= 1.0 Training accuracy: {kun.score(X_train, y_train):0.4f}')


# In[1553]:


1 - y_train.sum() / y_train.shape[0]


# In[1554]:


X_val = scaler.transform(X_val)
X_val = imputer.transform(X_val)
y_pred = kun.predict(X_val)
print(f'Balanced accuracy (train): {metrics.balanced_accuracy_score(y_val, y_pred):0.4f}')


# In[1555]:


print('LR with C = 1.0 Classification Report\n')
print(classification_report(y_val, y_pred))


# In[1556]:


# C-param changed and C-param is inversely propotional to accuracy

kun1 = LogisticRegression(solver='liblinear', max_iter=1000, C=0.5)
kun1.fit(X_train, y_train)


# In[1557]:


print(f'LR with C= 0.5 Training accuracy: {kun1.score(X_train, y_train):0.4f}')


# In[1558]:


1 - y_train.sum() / y_train.shape[0]


# In[1560]:


X_val = scaler.transform(X_val)
X_val = imputer.transform(X_val)
y_pred = kun.predict(X_val)
print(f'Balanced accuracy (train): {metrics.balanced_accuracy_score(y_val, y_pred):0.4f}')


# In[1561]:


print('LR with C = 0.5 Classification Report\n')
print(classification_report(y_val, y_pred))


# In[1562]:


from sklearn.svm import SVC
kun1 = SVC(C=1.0, kernel='linear', gamma= 1)
kun1.fit(X_train, y_train)


# In[1564]:


print(f'SVM Training accuracy for C = 1 : {kun1.score(X_train, y_train):0.4f}')


# In[1565]:


1 - y_train.sum() / y_train.shape[0]


# In[1566]:


X_val = scaler.transform(X_val)
X_val = imputer.transform(X_val)
y_pred = kun1.predict(X_val)
print(f'Balanced accuracy (train): {metrics.balanced_accuracy_score(y_val, y_pred):0.4f}')


# In[1567]:


print('SVM with linear kernel Classification Report\n when C = 1')
print(classification_report(y_val, y_pred))


# In[1568]:


#AUC value calculation

#from sklearn .metrics import roc_auc_score
#auc = np.round(roc_auc_score(y_val, y_pred), 3)
#print("AUC for our data is {}". format(auc))
#plt.plot(auc)
#plt.show()


# In[1569]:


#ROC Curve

fpr, tpr, _ = metrics.roc_curve(y_val,  y_pred)
auc = metrics.roc_auc_score(y_val, y_pred)


print("AUC for our data is : ",auc)
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[1570]:


from sklearn.svm import SVC
kun2 = SVC(C = .1, kernel='poly', gamma= 1)
kun2.fit(X_train, y_train)


# In[1571]:


print(f'Training accuracy: {kun2.score(X_train, y_train):0.4f}')


# In[1572]:


1 - y_train.sum() / y_train.shape[0]


# In[1573]:


X_val = scaler.transform(X_val)
X_val = imputer.transform(X_val)
y_pred = kun2.predict(X_val)
print(f'Balanced accuracy (train): {metrics.balanced_accuracy_score(y_val, y_pred):0.4f}')


# In[1574]:


print('SVM with poly kernel Classification Report\n when C = 0.01')
print(classification_report(y_val, y_pred))


# In[1575]:


#AUC value calculation

#from sklearn .metrics import roc_auc_score
#auc = np.round(roc_auc_score(y_val, y_pred), 3)
#print("AUC for our data is {}". format(auc))
#plt.plot(auc)
#plt.show()


# In[1576]:


#ROC Curve

fpr, tpr, _ = metrics.roc_curve(y_val,  y_pred)
auc = metrics.roc_auc_score(y_val, y_pred)


print("AUC for our data is : ",auc)
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[1577]:


kun3 = SVC(C= 1.0, kernel='poly', gamma= 5)
kun3.fit(X_train, y_train)


# In[1578]:


print(f'Training accuracy: {kun3.score(X_train, y_train):0.4f}')


# In[1579]:


1 - y_train.sum() / y_train.shape[0]


# In[1580]:


X_val = scaler.transform(X_val)
X_val = imputer.transform(X_val)
y_pred = kun.predict(X_val)
print(f'Balanced accuracy (train): {metrics.balanced_accuracy_score(y_val, y_pred):0.4f}')


# In[1581]:


print('SVM with poly kernel Classification Report\n')
print(classification_report(y_val, y_pred))


# In[1582]:


#AUC value calculation

#from sklearn .metrics import roc_auc_score
#auc = np.round(roc_auc_score(y_val, y_pred), 3)
#print("AUC for our data is {}". format(auc))
#plt.plot(auc)
#plt.show()


# In[1583]:


#ROC Curve

fpr, tpr, _ = metrics.roc_curve(y_val,  y_pred)
auc = metrics.roc_auc_score(y_val, y_pred)


print("AUC for our data is : ",auc)
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

