#!/usr/bin/env python
# coding: utf-8

# In[21]:


#import module
import csv as csv
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import csv
import scipy.stats as stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import IPython.display as vs
import io
import visuals as vs
import IPython.display

# Load the dataset of Winequality-red and white csv files which has 13 attribute and 7000 rows.
# In[22]:


#Load the dataset from csv file and having a look on first few 5 rows
df = pd.read_csv('Winequality-red.csv')
df.head()


# In[23]:


print("Total number of red_wine data: {}".format(df))


# In[24]:


#Load the dataset from csv file and having a look on first few 5 rows
df1 = pd.read_csv('Winequality-white.csv')
df1.head()


# In[25]:


print("Total number of white_wine data: {}".format(df1))


# # Merging the dataset of Winequality-red and white datasets

# In[26]:


#Reading in the merged dataset and taking a look at the first 5 rows
df2= pd.merge(df1,df,on='quality')


# In[27]:


df2.head()


# In[28]:


print("Total number of WQ_wine data: {}".format(df2))


# In[29]:


#Info on rows and columns of dataframe
df2.info()


# In[30]:


#finding the missing values in dataframe
df2.isnull().any()


# The output shows that there are no columns are empty

# In[31]:


df2 = df.shape[0]
print("Total number of WQ_wine data: {}".format(df2))


# In[32]:


#Number of wines with quality rating above 6
df2_above_6 = df.loc[(df['quality'] > 6)]
n_above_6 = df2_above_6.shape[0]
print("Wines with review 7 and above: {}".format(n_above_6))


# In[33]:


#Number of wines with quality rating above 7
df2_below_5 = df.loc[(df['quality'] < 5)]
n_below_5 = df2_below_5.shape[0]
print("Wines with review less than 5: {}".format(n_below_5))


# In[14]:


# Number of wines with quality rating between 5 to 6
df2_between_5 = df.loc[(df['quality'] >= 5) & (df['quality'] <= 6)]
n_between_5 = df2_between_5.shape[0]
print("Wines with review 5 and 6: {}".format(n_between_5))


# In[15]:


# Percentage of wines with quality rating above 6
greater_percent = n_above_6*100/df2
print("Percentage of wines with quality 7 and above: {:.2f}%".format(greater_percent))


# In[16]:


# Some more additional data analysis
display(np.round(df.describe()))


# In[17]:


# Visualize skewed continuous features of original data
vs.distribution(df,"quality")


# The above graph decribe the average of wines quality,which clearly shows that there are few wines which are high quality and great testing, and very few which are not that good.

# In[18]:


# defining correlation matrix for highly correalated variables
correlation = df.corr()
plt.figure(figsize=(14, 12))
heatmap = sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")


# In the above correlation matrix, the positive values shows direct co-relationship between features. The higher the values, the stronger features are highly correalated with each other. That means, if one features increases, the other one also tends to increase, and vice-versa.
# 
# The square that have negative values show an inverse co-relationship. if the negative values increase, the more inversely proportional they are and they will be more blue. This means that if the value of one features is higher, the value of the other one gets lower.
# 
# Therefore, squares close to zero indicate almost no co- dependency between those sets of attributes.
