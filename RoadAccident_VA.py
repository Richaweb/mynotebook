#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv as csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy import stats
import scipy.stats as stats
import matplotlib as mpl
import io
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix


# In[2]:


#Reading in the data from a csv and taking a look at the first 5 rows
df = pd.read_csv('Accidents_data.csv')
df.head()


# In[3]:


#Reading in the data from a csv and taking a look at the first 4 rows
df1 = pd.read_csv('Casualties_correct_edit.csv')
df1.head()


# In[4]:


#Reading in the data from a csv and taking a look at the first 4 rows
df2 = pd.read_csv('Vehicles_correct_edit.csv')
df2.head()


# In[5]:


#Reading in the merged dataset and taking a look at the first 5 rows
UK_df= pd.merge(df,df1,on='Accident_Index')


# In[6]:


df.info()#Info on rows and columns of dataframe


# In[7]:


df.head()


# In[8]:


#Removing the missing values in dataframe which is not required
df.isnull().sum()


# In[9]:


df = df[:100000]


# In[10]:


df.shape #finding the row and number of columns in the dataframe


# In[11]:


#setting up the target variable
target_variable = ['Number_of_Casualties']
# get the column names
all_cols = set(df.columns)    
# ones you don't care about
other_cols = all_cols.difference(target_variable)   


# In[12]:


#tidy the dataframe with list of column and variables 
tidy_order = list(other_cols) + list(target_variable)
tidy_df = df[tidy_order]


# In[13]:


#defining the correaltion matrix for highly correlated variables 
def correlation_heatmap(tidy_df):
    correlations = tidy_df.corr()

    fig, ax = plt.subplots(figsize=(30,25))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=1.5, annot=True, cmap = "BuPu")
    plt.show()
    
correlation_heatmap(tidy_df)


# In[14]:


#Using correlation matrix with output variable
correlations = tidy_df.corr()
cor_target = abs(correlations['Number_of_Casualties'])


# In[15]:


#Selecting the highly correlated features
relevant_features = cor_target[cor_target >0.1]
relevant_features


# In[16]:


df[['Location_Northing_OSGR','Number_of_Vehicles','Longitude','Speed_limit','Urban_or_Rural_Area','Latitude']].corr()


# In[30]:


#dropping the columns from the output of corelation matrix
df = df.drop(['Did_Police_Officer_Attend_Scene_of_Accident','Location_Northing_OSGR'],axis=1)


# In[31]:


df.head()


# In[32]:


#plotting scatterplot to show the relationship between age od band of driver and journey purpose of driver 
plt.scatter(df['p'], df['Weather_Conditions'], alpha=0.25)
plt.title('Scatter plot pythonspot.com')
plt.xlabel('Light_Conditions')
plt.ylabel('Speed_limit')
plt.show()


# In[34]:


#Implementing the PCA component for the dimentionality reduction
#Code Reference from towardsdatascience 
from sklearn.preprocessing import StandardScaler
features = ['Light_Conditions', 'Weather_Conditions', 'Local_Authority_(District)', 'Urban_or_Rural_Area']
# Separating out the features
x = df.loc[:, features].values
x


# In[21]:


# Merge Day_of_Week.csv to dataset
df_context = pd.read_csv('contextCSVs/Day_of_Week.csv')
df = pd.merge(df,df_context , how='inner', left_on='Day_of_Week', right_on='code')
df.rename(columns={'label': 'Label_Day_of_Week'}, inplace=True)
df


# In[22]:


# Separating out the target
y = df.loc[:,['Label_Day_of_Week']].values
df.head()


# In[23]:


# Standardizing the features
x = StandardScaler().fit_transform(x)


# In[24]:


#Implementing the PCA component for the dimentionality reduction
#Code Reference from towardsdatascience 
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
principalDf.head()


# In[25]:


#Concatenating dataframe along the axis
finalDf = pd.concat([principalDf, df[['Label_Day_of_Week']]], axis = 1)
finalDf.head()


# In[26]:


#Reading the shape file 
finalDf.shape


# In[27]:


#Reading the label of Accident_Severity
finalDf.head()


# In[28]:


# Plotting of 2 dimensional data
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
colors = ['r', 'g', 'b',]
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Label_Day_of_Week'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


# In[29]:


pca.explained_variance_ratio_


# In[52]:


# Using the Original dataset 
dforiginal = df[["Speed_limit","Police_Force","Number_of_Casualties"]]
#dforiginal= pd.merge(dforiginal,df_context_Number_of_Casualties, how='inner', left_on='Number_of_Casualties', right_on='code')

#del dforiginal['code']
dforiginal


# In[36]:


# One-hot encode categorical features
features = pd.get_dummies(dforiginal)
features


# In[39]:


print('We have {} observations with {} variables.'.format(*features.shape))


# In[40]:


# Labels are the values we want to predict
labels = features.Accident_Severity

# Remove the labels from the features
features= features.drop('Number_of_Casualties', axis = 1)

# Saving feature names for later use
feature_list = list(features.columns)


# In[41]:


df_context_Number_of_Casualties


# In[42]:


rfc=RandomForestClassifier(random_state=42)


# In[43]:


param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}


# In[44]:


def grid_search_wrapper(refit_score='accuracy_score'):
    skf = StratifiedKFold(n_splits=10)
    grid_search = GridSearchCV(clf, param_grid, scoring=scorers, refit=refit_score,
                           cv=skf, return_train_score=True, n_jobs=-1)
    grid_search.fit(xtrain.values, ytrain.values)

# make the predictions
    ypred = grid_search.predict(xtest.values)

    print('Best params for {}'.format(refit_score))
    print(grid_search.best_params_)

# confusion matrix on the test data.
    print('\nConfusion matrix of Random Forest optimized for {} on the test data:'.format(refit_score))
    print(pd.DataFrame(confusion_matrix(ytest, ypred),
                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
    return grid_search


# In[45]:


# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.3, random_state = 6, stratify = labels)


# In[46]:


x_train = train_features
y_train = test_features
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)


# In[47]:


rf=RandomForestClassifier(random_state = 7)
rf.fit(train_features,train_labels)


# In[48]:


#Show the distribution is consistent between training and test sets
print('train labels class distribution')
print(train_labels.value_counts(normalize=True))
print('test labels class distribution')
print(test_labels.value_counts(normalize=True))


# In[49]:


# Train the model on training data
rf=RandomForestClassifier(random_state = 7)

import time
start_time1 = time.time()
rf.fit(train_features, train_labels);
end1 = time.time()
print("--- %s seconds to train baseline model ---" % (end1 - start_time1))


# In[50]:


start_time1pred = time.time()
ypredictb1=rf.predict(test_features)
end1pred = time.time()
print("--- %s seconds to fit baseline model---" % (end1pred - start_time1pred))


# In[51]:


print("--- %s Accuracy with 30 trees for baseline model ---" % accuracy_score(test_labels,ypredictb1).round(4))


# In[ ]:




