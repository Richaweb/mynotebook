#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import module
import csv as csv
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
import seaborn as sns
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


#/ls in {} contain {} words.format(filepath,word_count)
 
#notebook_markdown_word_count('PDS CW.ipynb')


# # Data Preparation
# Load the data
# The dataset contain three files accidents, vehicles and casualty annd some context files linked to accidents,vehicles and casualty

# In[3]:


#Reading in the data from a csv and taking a look at the first 5 rows
df = pd.read_csv('Accidents.csv')
df.head()


# In[5]:


#Reading in the data from a csv and taking a look at the first 4 rows
df1 = pd.read_csv('Casualties_correct_edit.csv')
df1.head()


# In[ ]:


#Reading in the data from a csv and taking a look at the first 4 rows
df2 = pd.read_csv('Vehicles_correct_edit.csv')
df2.head()


# # Merging the dataset of Uk car accident dataset - accidents, vehicle and casualties

# In[ ]:


#Reading in the merged dataset and taking a look at the first 5 rows
UK_df= pd.merge(df,df1,on='Accident_Index')


# In[ ]:


UK_df.head()


# In[ ]:


#Reading in the merged dataset and taking a look at the first 5 rows
df= pd.merge(UK_df,df2,on='Accident_Index')


# In[ ]:


df.head()


# In[ ]:


df.info()#Info on rows and columns of dataframe


# In[ ]:


df.head()


# In[ ]:


#finding the missing values in dataframe
#df.isnull().sum()


# In[ ]:


#Removing the missing values in dataframe which is not required
df.isnull().sum()


# In[ ]:


df = df[:100000]


# In[ ]:


df.shape #finding the row and number of columns in the dataframe


# # Exploratory data analysis
# 

# In[ ]:


#Reading the accident_severity.csv file from dfataframe
df_context_Accident_Severity = pd.read_csv('contextCSVs/Accident_Severity.csv')

df_merge_severity= pd.merge(df,df_context_Accident_Severity, how='inner', left_on='Accident_Severity', right_on='code')
#df = df.drop(['code'],axis=1)

df_merge_severity.rename(columns={'label': 'Severity'}, inplace=True)


# In[ ]:


#geometry = Point(xy) for xy in zip (df["Longitude"], df["Latitude"])]
#geometry[:3]


# In[ ]:


#geo_df = gpd.accidentsdf (# specify our data)
    #crs = crs,# specify our coordinate reference system
    #geometry = geometry
    #geo_df.head()


# In[ ]:


df_severity = df_merge_severity[["Severity", "Number_of_Casualties" , "Age_Band_of_Casualty"]]


# In[ ]:


#Age_Band.csv
df_context_Age_Band = pd.read_csv('contextCSVs/Age_Band.csv')
df_context_Age_Band.head() 
df_age_band_merge= pd.merge(df_severity,df_context_Age_Band, how='inner', left_on='Age_Band_of_Casualty', right_on='code')

df_age_band_merge
df_age_band_merge.rename(columns={'label': 'Label_Age_Band_of_Casualty'}, inplace=True)
df_age_band = df_age_band_merge[["Severity", "Number_of_Casualties" , "Label_Age_Band_of_Casualty"]]

df_age_band.head()


# In[ ]:


#Plotting a bar chart below on the basis of accident severity column
sns.set_color_codes("muted")
sns.set(rc={'figure.figsize':(8,6)})
sns.set_style("ticks")
sns.countplot(df_age_band['Severity'])
plt.title('Count Plot of Accidents')
plt.xlabel('Severity of Accident')
plt.ylabel('Count of Accident')
sns.despine(right = True, top = True)


# This barchart represent the three severity category of, count of accidents serious, slight and fatal in the X axis and count of accident in the Y axis.In the severity of accident, slight is the highest followed by serious and fatal. Slight severity accident which are more than 40000 whereas Serious accident fall under 10000 and fatal accidents are even lesser than serious and slight.

# In[ ]:


#Calculating the overall success and failure rates and those for other 'states'
Percent_Serious=round(df_age_band[df_age_band['Severity']=='Serious']['Severity'].value_counts()/len(df_age_band['Severity'])*100,2)
Percent_Slight=round(df_age_band[df_age_band['Severity']=='Slight']['Severity'].value_counts()/len(df_age_band['Severity'])*100,2)
Percent_Fatal=round(df_age_band[df_age_band['Severity']=='Fatal']['Severity'].value_counts()/len(df_age_band['Severity'])*100,2)
SFC = float(Percent_Fatal) + float(Percent_Slight) + float(Percent_Serious)
other = round(100-SFC,2)
print('Serious Accident rate is',Percent_Serious[0],'%')
print('Slight Accident rate is',Percent_Slight[0],'%') 
print('Fatal  Accident is',Percent_Fatal[0],'%') 
print('The percentage of Severity of Accidents which are not categorised',other,'%')


# In[ ]:


#Creating a stacked bar chart of accident by severity, displayed by Age band group
import matplotlib as mpl
mpl.style.use('seaborn-white')
dfstates=df_age_band.groupby(['Label_Age_Band_of_Casualty','Severity']).size().fillna(0).unstack() #Creating the count dataframe
dfstates=dfstates.divide(dfstates.iloc[:,:].sum(axis=1),axis=0)*100 #Calculating percentage values of the severity of each Age band group
dfstates.sort_values(by=['Label_Age_Band_of_Casualty']).plot(kind='bar',stacked=True, label='% of state by category') #creating a stacked bar chart
plt.legend(loc=(1.05,0.75))
plt.title('Percentage of Accidents in each Severity, by Driver Age Band Group')
plt.xlabel('Casualty Age-Band Group')
plt.ylabel('Percent of Accident')


# An intial overview of the category of projects is shown above on the basis of driver age band group and percent of accident. In the above bar chart I am analysing the percentage of accidents along with the severity of causalities for each casualty age group. I can clearly see the fatality is the highest for the over 66-70 and 75 age group and the least is 36-45 age group.
#  

# In[ ]:


# load Casualty_Type.csv
df_context_Casualty_Type = pd.read_csv('contextCSVs/Casualty_Type.csv')

df_context_casualty_type_merge = pd.merge(df,df_context_Casualty_Type, how='inner', left_on='Casualty_Type', right_on='code')
df_context_casualty_type_merge.rename(columns={'label': 'Label_Casualty_Type'}, inplace=True)


df_context_casualty_type_merge = df_context_casualty_type_merge[['Label_Casualty_Type', 'Number_of_Casualties']]


# In[ ]:


#Merge and Get number of Casualties group by Casualty type 
groupbycategory_casualty = df_context_casualty_type_merge.groupby('Label_Casualty_Type')['Number_of_Casualties'].sum()
groupbycategory_casualty.round()


# In[ ]:


#Preparing and Plotting the Pie chart
d = groupbycategory_casualty.sort_values(ascending=False)
d.to_dict()
from operator import itemgetter
# sort by value descending
items_sorted = sorted(d.items(), key=itemgetter(1), reverse=True)
# calculate sum of others
others = ('Other', sum(map(itemgetter(1), items_sorted[8:])))
# construct dictionary
d = dict([*items_sorted[:8], others])
print(d)
listkeys = d.keys()
listvalues = list(d.values())
mpl.style.use('bmh')

fig1, ax1 = plt.subplots(figsize=(10, 10))

ax1.pie(listvalues, labels=listkeys, autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.title('% of Total Casualities by Casualty Type')
plt.show()


# The Pie chart represent percentage of causalities involving various category of Casualty type. The highest percentage of casuality is Car Accident which is 68.3%. This is followed by 6% which is for pedestrian.

# In[ ]:


# Merge Light_Conditions.csv to dataset
df_context_light_condition = pd.read_csv('contextCSVs/Light_Conditions.csv')
df_merge_light_condition= pd.merge(df,df_context_light_condition, how='inner', left_on='Light_Conditions', right_on='code')
df_merge_light_condition.rename(columns={'label': 'Label_LightCondition'}, inplace=True)
df_merge_light_condition
df_light_condition = df_merge_light_condition[[ "Number_of_Casualties" , "Label_LightCondition"]]
df_light_condition
sns.set_color_codes("muted")
sns.set(rc={'figure.figsize':(16,8)})
sns.set_style("ticks")
sns.countplot(df_light_condition['Label_LightCondition'])
plt.title('Count Plot of Accidents')
plt.xlabel('Light condition')
plt.ylabel('Count of Accident')
sns.despine(right = True, top = True)


# The bar chart represent number of count of accidents based on different lighting condition. As you can see that in the daylight on the light condition x axis maximun accident has happened and the least has happened in the darkness- lighting unknown.

# In[ ]:


# Merge Road_Type.csv to dataset
df_context = pd.read_csv('contextCSVs/Road_Type.csv')
df_merge = pd.merge(df,df_context , how='inner', left_on='Road_Type', right_on='code')
df_merge.rename(columns={'label': 'Label_Road_Type'}, inplace=True)
df_merge
df_merge_final = df_merge[[ "Number_of_Casualties" , "Label_Road_Type"]]
df_merge_final
sns.set_color_codes("muted")
sns.set(rc={'figure.figsize':(20,8)})
sns.set_style("ticks")
sns.countplot(df_merge_final['Label_Road_Type'])
plt.title('Count Plot of Accidents')
plt.xlabel('Road Type Where Accident Took Place')
plt.ylabel('Count of Accident')
sns.despine(right = True, top = True)


# In[ ]:


# Merge Vehicle_Manoeuvre.csv to dataset
df_context = pd.read_csv('contextCSVs/Vehicle_Manoeuvre.csv')
df_merge = pd.merge(df,df_context , how='inner', left_on='Vehicle_Manoeuvre', right_on='code')
df_merge.rename(columns={'label': 'Label_Vehicle_Manoeuvre'}, inplace=True)
df_merge
df_merge_final = df_merge[[ "Number_of_Casualties" , "Label_Vehicle_Manoeuvre"]]
df_merge_final
sns.set_color_codes("muted")
sns.set(rc={'figure.figsize':(30,8)})
sns.set_style("ticks")
sns.countplot(df_merge_final['Label_Vehicle_Manoeuvre'])
plt.title('Count Plot of Accidents')
plt.xlabel('Vehicle Manoeuvre resulted in Accident')
plt.ylabel('Count of Accident')
sns.despine(right = True, top = True)


# This bar chart shows number of accidents vs Vehicle Manoeuvre resulted in Accident. Vechicle moving ahead had the mximum number of accidents.

# In[ ]:


# Merge Ped_Movement.csv to dataset
df_context = pd.read_csv('contextCSVs/Ped_Movement.csv')
df_merge = pd.merge(df,df_context , how='inner', left_on='Pedestrian_Movement', right_on='code')
df_merge.rename(columns={'label': 'Label_Ped_Movement'}, inplace=True)
df_merge
df_merge_final = df_merge[[ "Number_of_Casualties" , "Label_Ped_Movement"]]
df_merge_final
sns.set_color_codes("muted")
sns.set(rc={'figure.figsize':(30,8)})
sns.set_style("ticks")
sns.countplot(df_merge_final['Label_Ped_Movement'])
plt.title('Count Plot of Accidents')
plt.xlabel('Pedestrian Movement When Accident Took Place')
plt.ylabel('Count of Accident')
sns.despine(right = True, top = True)


# This bar chart shows number of accidents vs Pedestrian Movement when Accident took place.

# In[ ]:


#Accident_Severity.csv
df_context_Accident_Severity = pd.read_csv('contextCSVs/Accident_Severity.csv')

df= pd.merge(df,df_context_Accident_Severity, how='inner', left_on='Accident_Severity', right_on='code', suffixes=('_left_severity', '_right_severity'))

df.head()


# In[ ]:


#creating a scatterplot to find the outliers in the casulaties dataset in the Age_of_Casualty and sex_of_casualty column
f, ax = plt.subplots(1,2)
#fig, ax = plt.subplots(figsize=(16,8))
sns.scatterplot(x='Age_of_Casualty', y='Sex_of_Casualty', data=df1, palette='hls', ax=ax[0], color = 'darkorange')
sns.scatterplot(x='Age_of_Casualty', y='Sex_of_Casualty', data=df1, palette='hls', alpha=0.1, ax=ax[1], color='turquoise')


# I have used a scatterplot to represent two different numerical variables.The position of each dot on the horizontal and vertical axis indicates values for an individual data point.Y axis is represnting sex of casulaty where -1 is gender unknown and 1 is male and 2 represnts female. X axis represents the age of casualty. The graph clearly shows the people of age band 25 to 40 have more casualities as compared to the people over 75. There were equal casualities of male and female. There were lot casualities where gender were unknown.

# In[ ]:


sns.boxplot(x=df["Number_of_Casualties"], color = 'mediumslateblue')


# A box plot is a method for graphically representing groups of numerical data through their quartiles. Box plots may also have lines extending vertically from the boxes and whiskers indicating variability outside the upper and lower quartiles.Outliers are plotted as individual points. Therefore in the diagram above outliers define from number 4 to 20 in the number of Casualties.

# There is very low data about the severity of casuality where the age of casualty is over age of 70, this will have no significance for my objective and so I am droping the rows of dataset where age of casualty is greater than 70.

# In[ ]:


#dropping the rows which are greater than 70
df.drop(df[df['Age_of_Casualty'] > 70].index, inplace = True)


# In[ ]:


#showing the 15 rows and 64 columns 
(df.head(15))


# In[ ]:


df.head()


# Let's find out the correaltion matrix of the dataframe which are highly correlated.

# In[ ]:


#setting up the target variable
target_variable = ['Accident_Severity']
# get the column names
all_cols = set(df.columns)    
# ones you don't care about
other_cols = all_cols.difference(target_variable)    


# In[ ]:


#tidy the dataframe with list of column and variables 
tidy_order = list(other_cols) + list(target_variable)
tidy_df = df[tidy_order]


# In[ ]:


#defining the correaltion matrix for highly correlated variables 
def correlation_heatmap(tidy_df):
    correlations = tidy_df.corr()

    fig, ax = plt.subplots(figsize=(30,25))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=1.5, annot=True, cmap = "BuPu")
    plt.show()
    
correlation_heatmap(tidy_df)


# In[ ]:


#Using correlation matrix with output variable
correlations = tidy_df.corr()
cor_target = abs(correlations['Accident_Severity'])


# In[ ]:


#Selecting the highly correlated features
relevant_features = cor_target[cor_target >0.1]
relevant_features


# Accident_Severity is the highly correlated variable in the correlation matrix which is shown above.

# In[ ]:


df[["Pedestrian_Location","Casualty_Severity","Speed_limit","Urban_or_Rural_Area","Number_of_Casualties","Vehicle_Manoeuvre",'Casualty_Reference']].corr()


# Urban_or_Rural_Area and Casualty_Reference are less likely correlated variable therefore I am droping the columns from the output correlation matrix

# In[ ]:


#dropping the columns from the output of corelation matrix
df = df.drop(['Urban_or_Rural_Area','Casualty_Reference'],axis=1)


# In[ ]:


df.head()


# In[ ]:


#plotting scatterplot to show the relationship between age od band of driver and journey purpose of driver 
plt.scatter(df['Age_Band_of_Driver'], df['Journey_Purpose_of_Driver'], alpha=0.15)
plt.title('Scatter plot pythonspot.com')
plt.xlabel('Age_Band_of_Driver')
plt.ylabel('Journey_Purpose_of_Driver')
plt.show()


# The scatterplot above shows the scatter representation of age band of driver and journey purpose of driver age band of driver which means the driver between the age of 20 to 60 were driving the car for the journey purpose on the Y axis under 2 and 4 were  commuting to and from work or taking kids to/from school. Also all the Age Band of Driver purpose of journey ( ~Y axis- integer value 15 ) was unknown.

# In[ ]:


#Plotting the line plot to show below
au = sns.lineplot(x="Age_of_Driver", y="Sex_of_Driver", data=df)


# The Line plot above show the age of driver on X axis and sex of driver on Y axis which represent that at what age of driver are more prone to do accident and what genders are most likely to do accidents.

# # Computation Modeling

# In[ ]:


#Implementing the PCA component for the dimentionality reduction
#Code Reference from towardsdatascience 
from sklearn.preprocessing import StandardScaler
features = ['Police_Force', 'Accident_Severity', 'Number_of_Vehicles', 'Number_of_Casualties']
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['Accident_Severity']].values
df.head()


# In[ ]:


# Standardizing the features
x = StandardScaler().fit_transform(x)


# In[ ]:


#Implementing the PCA component for the dimentionality reduction
#Code Reference from towardsdatascience 
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
principalDf.head()


# In[ ]:


#Representating the dimentionality of the dataframe
#Code Reference from towardsdatascience
principalDf.shape


# In[ ]:


#Concatenating dataframe along the axis
finalDf = pd.concat([principalDf, df[['label']]], axis = 1)
finalDf.head()


# In[ ]:


#Reading the shape file 
finalDf.shape


# In[ ]:


#Reading the label of Accident_Severity
df_context_Accident_Severity.head()


# In[ ]:


# Plotting of 2 dimensional data
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Slight', 'Serious', 'Fatal']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['label'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


# In[ ]:


pca.explained_variance_ratio_


# Generally, it is preferred that at least 95% of the variance is captured by the PCA within a training set in order to use in a regression or classification algorithm. In terms of the limitations of PCA, it is not scale invariant. Despite scaling the numeric variables, the use of dummy variables does not provide meaningful output for dimension reduction here.
# 
# By using the attribute explained_variance_ratio, I can see that the first principal component contains 33.07% of the variance and the second principal component contains 26.12% of the variance. Together, the two components contain 59.19% of the information.

# # Random Forest Model
# 
# I choose upon using random forest for classification as it is generally considered to be robust method due to the process of averaging all predictions, which removes biases and decreases chances of overfitting.
# 
# The Random Forest algorithm operates by selecting samples from a dataset at random and generating decision trees for each of those samples. The algorithm then takes the majority of the trees’ decisions as the final choice.
# Random Forests are useful for classification tasks as they do not require as much pre-processing as other approaches  and accept both categorical and numerical variables.
# 
# Random Forests also provide a relative ranking of feature importance, which facilitates feature selection.

# In[ ]:


# Using the Original dataset 
dforiginal = df[["Accident_Severity","Police_Force","Number_of_Casualties"]]
dforiginal= pd.merge(dforiginal,df_context_Accident_Severity, how='inner', left_on='Accident_Severity', right_on='code')

del dforiginal['code']
dforiginal


# I have chosen features number of casualities (column - Number_of_Casualties) and severity (Accident_Severity) based on the impact of the accident  and quick was the police to respond which is represented by the column "Police_Force". The Category chosen is the label of severity of accident.

# In[ ]:


# One-hot encode categorical features
features = pd.get_dummies(dforiginal)
features


# In[ ]:


print('We have {} observations with {} variables.'.format(*features.shape))


# In[ ]:


# Labels are the values we want to predict
labels = features.Accident_Severity

# Remove the labels from the features
features= features.drop('Accident_Severity', axis = 1)

# Saving feature names for later use
feature_list = list(features.columns)


# In[ ]:


df_context_Accident_Severity


# In[ ]:


rfc=RandomForestClassifier(random_state=42)


# In[ ]:


param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}


# In[ ]:


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


# In the case of a random forest hyperparameters include the number of decision trees in the forest and the number of features considered by each tree when splitting a node. The parameters of a random forest are the variables and thresholds used to split each node learned during training.

# In[ ]:


# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.3, random_state = 6, stratify = labels)


# In[ ]:


x_train = train_features
y_train = test_features
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)


# In[ ]:


rf=RandomForestClassifier(random_state = 7)
rf.fit(train_features,train_labels)


# In[ ]:


#Show the distribution is consistent between training and test sets
print('train labels class distribution')
print(train_labels.value_counts(normalize=True))
print('test labels class distribution')
print(test_labels.value_counts(normalize=True))


# In[ ]:


# Train the model on training data
rf=RandomForestClassifier(random_state = 7)

import time
start_time1 = time.time()
rf.fit(train_features, train_labels);
end1 = time.time()
print("--- %s seconds to train baseline model ---" % (end1 - start_time1))


# In[ ]:


start_time1pred = time.time()
ypredictb1=rf.predict(test_features)
end1pred = time.time()
print("--- %s seconds to fit baseline model---" % (end1pred - start_time1pred))


# In[ ]:


print("--- %s Accuracy with 30 trees for baseline model ---" % accuracy_score(test_labels,ypredictb1).round(4))


# #Confusion matrix
# A confusion matrix is a table that is often used to describe the performance of a classification model on a set of test data for which the true values are known. It allows the visualization of the performance of an algorithm.
# 
# It allows easy identification of confusion between classes.One class is commonly mislabeled as the other.Most performance measures are computed from the confusion matrix.Below confusion Matrix is representing accuracy score on predicted label and true label.

# In[ ]:


# Confusion Matrix on test data
print(confusion_matrix(test_labels,ypredictb1))
print(classification_report(test_labels,ypredictb1))
cm = confusion_matrix(test_labels,ypredictb1)
plt.matshow(cm)
data = np.random.random((5,5))
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

