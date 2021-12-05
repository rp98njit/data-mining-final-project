#!/usr/bin/env python
# coding: utf-8

# # Data Mining Final Project
# ##           - Rajendra Prasad Patil
# 
# ### Glossary:
# * Import libraries
# * Load dataset
# * Analysis on dataset
# * Splitting the dataset into labels and features
# * Performing normalization on dataset
# * Splitting dataset using K fold 
# * Running the model
#     * SVM Model
#     * K Nearest Neighbors
#     * Random Forest Classifier
# * Output Performance Metrics
# 
# 

# ### Importing libraries

# In[1]:


import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

# SVM classifier
from sklearn import svm

# KNN classifier
from sklearn.neighbors import KNeighborsClassifier 

#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

# Import libraries for lstm classification
from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential

# for checking the model accuracy
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# ### Loading the dataset

# In[2]:


dataset = load_breast_cancer()


# In[3]:


input_length = len(dataset['data'][0])


# ### Preliminary analysis

# In[4]:


class_names = dataset['target_names']
print('Target variables  : ', class_names)

(unique, counts) = np.unique(dataset['target'], return_counts=True)

print('Unique values of the target variable', unique)
print('Counts of the target variable :', counts)


# * The dataset is suited for binary classification
# * The dataset has no skewed nature

# ### The data is split into features and labels

# In[5]:


X = dataset['data']
y = dataset['target']


# ### Apply normalization operation for numerical stability

# In[6]:


standardizer = StandardScaler()
X = standardizer.fit_transform(X)


# # Performance Metrics
# Function to calculate all the available performance metrics

# In[7]:


performance_metrics = ['True Negative', 'False Positive', 'False Negative', 'True Positivity', 'Sensitivity', 'Specificity', 
                       'Precision', 'Accuracy', 'F1 Score', 'Error Rate', 'Negative Predicted Value', 'False Positve Rate', 
                       'False Discovery Rate', 'False Negative Rate', 'Balanced Accuracy', 'True Skill Statistics', 
                       'Heidke Skill Score']

def compute_performance_metrics(prediction, y_test, df, is_lstm = False):
    
    if is_lstm:
        threshold = 0.80
        for i, each in enumerate(prediction):
            if each[0] > threshold:
                prediction[i] = 1
            else:
                prediction[i] = 0
    
    TN, FP, FN, TP = confusion_matrix(y_test, prediction).ravel()
    
    sensitivity = TP / (TP + FN)
    specificity = TN / (FP + TN)
    precision = TP / (TP + FP)
    accuracy =  (TP+TN) /(TP+FP+TN+FN)
    f1_score = 2 * TP / ((2 * TP) + FP + FN)
    error_rate = (FP + FN) / (TP + FP + FN + TN)
    negative_predicted_value = TN / (TN + FN)
    false_positive_rate = FP / (FP + TN)
    false_discovery_rate = FP / (FP + TP)
    false_negative_rate = FN / (FN + TP)
    balanced_accuracy = 0.5 * ((TP / (TP + FN)) + (TN / (TN + FP)))
    true_skill_statistics = ((TP / (TP + FN)) - (FP / (TN + FP)))
    heidke_skill_score = 2 * ((TP * TN) - (FP * FN)) / (((𝑇𝑃 + 𝐹𝑁) * (𝐹𝑁 + 𝑇𝑁)) +((TP+FP) * (𝐹𝑃 + 𝑇𝑁)))
    
    df = df.append({performance_metrics[0]: TN, performance_metrics[1]: FP, performance_metrics[2]: FN, 
                    performance_metrics[3]: TP, performance_metrics[4]: sensitivity, performance_metrics[5]: specificity, 
                    performance_metrics[6]: precision, performance_metrics[7]: accuracy, performance_metrics[8]: f1_score, 
                    performance_metrics[9]: error_rate, performance_metrics[10]: negative_predicted_value, 
                    performance_metrics[11]: false_positive_rate, performance_metrics[12]: false_discovery_rate, 
                    performance_metrics[13]: false_negative_rate, performance_metrics[14]: 
                    balanced_accuracy, performance_metrics[15]: true_skill_statistics,
                    performance_metrics[16]: heidke_skill_score}, ignore_index=True)
    return df    


# # K-fold cross validation

# In[8]:


from sklearn.model_selection import KFold
kfold = KFold(n_splits=10, shuffle=True, random_state=0)


# ### Dataframes for performance metrics

# In[9]:


svm_metrics_df = pd.DataFrame(columns=performance_metrics)
kn_metrics_df = pd.DataFrame(columns=performance_metrics)
rf_metrics_df = pd.DataFrame(columns=performance_metrics)
lstm_metrics_df = pd.DataFrame(columns=performance_metrics)


# ## SVM Model

# In[10]:


svm_model = svm.SVC()
for train_index, test_index in kfold.split(X):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

    # we train the algorithm with training data and training output
    svm_model.fit(X_train, y_train)

    # we pass the testing data to the stored algorithm to predict the outcome
    prediction = svm_model.predict(X_test)

    # print metrics
    svm_metrics_df = compute_performance_metrics(prediction, y_test, svm_metrics_df)

svm_metrics_df.index += 1
svm_metrics_df.loc['Average'] = svm_metrics_df.mean()


# In[11]:


svm_metrics_df


# ### K-Nearest Neighbors

# In[12]:


model = KNeighborsClassifier(n_neighbors=3) # this examines 3 neighbors for putting the data into class

for train_index, test_index in kfold.split(X):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

    # we train the algorithm with training data and training output
    model.fit(X_train, y_train)

    # we pass the testing data to the stored algorithm to predict the outcome
    prediction = model.predict(X_test)

    # print metrics
    kn_metrics_df = compute_performance_metrics(prediction, y_test, kn_metrics_df)

kn_metrics_df.index += 1
kn_metrics_df.loc['Average'] = kn_metrics_df.mean()


# In[13]:


kn_metrics_df


# ### Random Forest Classifier

# In[14]:


#Create a Gaussian Classifier
model = RandomForestClassifier(n_estimators=100)

for train_index, test_index in kfold.split(X):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

    # we train the algorithm with training data and training output
    model.fit(X_train, y_train)

    # we pass the testing data to the stored algorithm to predict the outcome
    prediction = model.predict(X_test)

    # print metrics
    rf_metrics_df = compute_performance_metrics(prediction, y_test, rf_metrics_df)

rf_metrics_df.index += 1
rf_metrics_df.loc['Average'] = rf_metrics_df.mean()


# In[15]:


rf_metrics_df


# # LSTM classifier

# In[16]:


model = Sequential()
model.add(LSTM(20, input_shape=(input_length, 1)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

for train_index, test_index in kfold.split(X):
    print('*'*100)
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

    # we train the algorithm with training data and training output
    model.fit(X_train, y_train, batch_size=8, epochs=10, validation_data=(X_test, y_test), verbose = 1)

    # we pass the testing data to the stored algorithm to predict the outcome
    prediction = model.predict(X_test)
    
    # print metrics
    lstm_metrics_df = compute_performance_metrics(prediction, y_test, lstm_metrics_df, is_lstm=True)

lstm_metrics_df.index += 1
lstm_metrics_df.loc['Average'] = lstm_metrics_df.mean()


# In[17]:


lstm_metrics_df


# ### Cumulative metrics

# In[29]:


all_dfs = [svm_metrics_df, kn_metrics_df, lstm_metrics_df]
all_names = ['SVM', 'KNN', 'LSTM']


# #### 1st Fold

# In[34]:


# lstm_metrics_df.loc[1,:]
df = pd.DataFrame(columns=performance_metrics)
fold_count = 1
for i, each_df in enumerate(all_dfs):
    temp_df = each_df.xs(fold_count)
    temp_df.name = all_names[i]
    df = df.append(temp_df)
df


# #### 2nd Fold

# In[35]:


# lstm_metrics_df.loc[1,:]
df = pd.DataFrame(columns=performance_metrics)
fold_count = 2
for i, each_df in enumerate(all_dfs):
    temp_df = each_df.xs(fold_count)
    temp_df.name = all_names[i]
    df = df.append(temp_df)
df


# #### 3rd Fold

# In[36]:


# lstm_metrics_df.loc[1,:]
df = pd.DataFrame(columns=performance_metrics)
fold_count = 3
for i, each_df in enumerate(all_dfs):
    temp_df = each_df.xs(fold_count)
    temp_df.name = all_names[i]
    df = df.append(temp_df)
df


# #### 4th Fold

# In[37]:


# lstm_metrics_df.loc[1,:]
df = pd.DataFrame(columns=performance_metrics)
fold_count = 4
for i, each_df in enumerate(all_dfs):
    temp_df = each_df.xs(fold_count)
    temp_df.name = all_names[i]
    df = df.append(temp_df)
df


# #### 5th Fold

# In[38]:


# lstm_metrics_df.loc[1,:]
df = pd.DataFrame(columns=performance_metrics)
fold_count = 5
for i, each_df in enumerate(all_dfs):
    temp_df = each_df.xs(fold_count)
    temp_df.name = all_names[i]
    df = df.append(temp_df)
df


# #### 6th Fold

# In[39]:


# lstm_metrics_df.loc[1,:]
df = pd.DataFrame(columns=performance_metrics)
fold_count = 6
for i, each_df in enumerate(all_dfs):
    temp_df = each_df.xs(fold_count)
    temp_df.name = all_names[i]
    df = df.append(temp_df)
df


# #### 7th Fold

# In[40]:


# lstm_metrics_df.loc[1,:]
df = pd.DataFrame(columns=performance_metrics)
fold_count = 7
for i, each_df in enumerate(all_dfs):
    temp_df = each_df.xs(fold_count)
    temp_df.name = all_names[i]
    df = df.append(temp_df)
df


# #### 8th Fold

# In[42]:


# lstm_metrics_df.loc[1,:]
df = pd.DataFrame(columns=performance_metrics)
fold_count = 8
for i, each_df in enumerate(all_dfs):
    temp_df = each_df.xs(fold_count)
    temp_df.name = all_names[i]
    df = df.append(temp_df)
df


# #### 9th Fold

# In[43]:


# lstm_metrics_df.loc[1,:]
df = pd.DataFrame(columns=performance_metrics)
fold_count = 9
for i, each_df in enumerate(all_dfs):
    temp_df = each_df.xs(fold_count)
    temp_df.name = all_names[i]
    df = df.append(temp_df)
df


# #### 10th Fold

# In[44]:


# lstm_metrics_df.loc[1,:]
df = pd.DataFrame(columns=performance_metrics)
fold_count = 10
for i, each_df in enumerate(all_dfs):
    temp_df = each_df.xs(fold_count)
    temp_df.name = all_names[i]
    df = df.append(temp_df)
df


# #### Average of all

# In[45]:


# lstm_metrics_df.loc[1,:]
df = pd.DataFrame(columns=performance_metrics)
fold_count = 'Average'
for i, each_df in enumerate(all_dfs):
    temp_df = each_df.xs(fold_count)
    temp_df.name = all_names[i]
    df = df.append(temp_df)
df


# ### Observation
# * I consider balanced accuracy to be the optimal metric to find the best model.
# * The case being, SVM is the model which is giving the highest balanced accuracy.
# 
# ### Why is SVM performing better?
# * SVM doesn't get affected by outliers
# * It does not suffer from overfitting
# * It is more efficient than other ML algorithms listed here
