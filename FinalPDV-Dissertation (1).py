#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import module
import csv as csv
import datetime as dt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O 
import matplotlib.pyplot as plt
import plotly.express as px
import os
import seaborn as sns
import csv
from pandas import Timestamp
import datetime as dt
import folium
from datetime import datetime
from folium.plugins import HeatMapWithTime
import folium.plugins as plugins


# # Data Preparation

# Loading the motor vehicle collisions dataset which is reported by new york police department in 2020. The dataset is consist of three csv files - crashes, vehicles and person. The motor vehicles collisions_crashes files contain information about contributing factor and vehicle type which were involved in the road traffic incidents. 

# In[2]:


#Reading in the data from a csv and taking a look at the first 5 rows
mvc_crashes = pd.read_csv('Motor_Vehicle_Collisions_-_Crashes.csv', low_memory=False)
mvc_crashes.head()


# In[3]:


#dropping columns which has null value.
mvc_crashes= mvc_crashes.drop(['CONTRIBUTING FACTOR VEHICLE 3','CONTRIBUTING FACTOR VEHICLE 4','CONTRIBUTING FACTOR VEHICLE 5'], axis=1)


# Loading the motor vehicles collision_vehicles csv files to and reading the first five cloumns of the dataset. The vehicles csv files contain various vehicles information like model, make, type, point of impact etc.

# In[4]:


#Reading in the data from a csv and taking a look at the first 5 rows
mvc_vehicles = pd.read_csv('Motor_Vehicle_Collisions_-_Vehicles.csv', low_memory=False)
mvc_vehicles.head()


# In[5]:


#dropping columns which has null value.
mvc_vehicles= mvc_vehicles.drop(['PUBLIC_PROPERTY_DAMAGE_TYPE'],axis=1)


# Loading the motor vehicle collisions_person.csv files which elaborate the coloumn like person_injury, person_type, contributing_factors etc and reading the first five rows.

# In[6]:


#Reading in the data from a csv and taking a look at the first 5 rows
mvc_persons = pd.read_csv('Motor_Vehicle_Collisions_-_Person.csv')
mvc_persons.head()


# In[7]:


# dropping columns which has null value.
mvc_persons= mvc_persons.drop(['CONTRIBUTING_FACTOR_1','CONTRIBUTING_FACTOR_2'],axis=1)


# Merging all the three dataset like vehicle, crashes and person files to make a one dataset.

# In[8]:


#Reading in the merged dataset and taking a look at the first 5 rows
mvc_crashes_vehicles_persons = mvc_crashes.merge(mvc_vehicles,on='COLLISION_ID').merge(mvc_persons,on='COLLISION_ID')
mvc_crashes_vehicles_persons.head()


# # Data Cleaning
Before starting the data analysis we need to make sure the data is clean. for instance, we should check for:

1.Duplicate records
2.Consistent formatting
3.Missing values
4.Obviously wrong values (x)
# Before starting the data analysis we need to make sure the data is clean. for instance, we should check for:
# Duplicate records

# In[9]:


#Returns a Series with True and False values that describe which rows in the Dataset are duplicated and not
dup_rows = mvc_crashes_vehicles_persons.duplicated().sum()
dup_rows


# As we can see in the above code no dupliacte row were found therefore there is no duplicates in the mvc_crashes_vehicles_persons dataset.

# In[10]:


#finding the missing values
mvc_crashes_vehicles_persons.isnull().sum()


# As we can see the above missing values, there are column such as borough, zip code, latitude, ped location etc has got a missing values however, I cannot delete this row because they are linked to other column which has got useful data to represent according to the research objectives.

# Consistent formatting equally important for data anaylsis

# In[11]:


#Converting the columns to required data type from object to strings 
mvc_crashes_vehicles_persons['VEHICLE TYPE CODE 1']= mvc_crashes_vehicles_persons['VEHICLE TYPE CODE 1'].astype("string")
mvc_crashes_vehicles_persons['VEHICLE TYPE CODE 2']= mvc_crashes_vehicles_persons['VEHICLE TYPE CODE 2'].astype("string")
mvc_crashes_vehicles_persons['VEHICLE TYPE CODE 3']= mvc_crashes_vehicles_persons['VEHICLE TYPE CODE 3'].astype("string")
mvc_crashes_vehicles_persons['VEHICLE TYPE CODE 4']= mvc_crashes_vehicles_persons['VEHICLE TYPE CODE 4'].astype("string")
mvc_crashes_vehicles_persons['VEHICLE TYPE CODE 5']= mvc_crashes_vehicles_persons['VEHICLE TYPE CODE 5'].astype("string")

mvc_crashes_vehicles_persons['BOROUGH']= mvc_crashes_vehicles_persons['BOROUGH'].astype("string")

mvc_crashes_vehicles_persons['LOCATION']= mvc_crashes_vehicles_persons['LOCATION'].astype("string")

#Converting the columns to required data type from float to integers
mvc_crashes_vehicles_persons['NUMBER OF PERSONS INJURED'] = mvc_crashes_vehicles_persons['NUMBER OF PERSONS INJURED'].fillna(0.0).astype(int)
mvc_crashes_vehicles_persons['NUMBER OF PERSONS KILLED'] = mvc_crashes_vehicles_persons['NUMBER OF PERSONS KILLED'].fillna(0.0).astype(int)
mvc_crashes_vehicles_persons['CRASH DATE'] = pd.to_datetime(mvc_crashes_vehicles_persons['CRASH DATE'])


# While reading from csv file using panda, the column which are off string type in csv get stored as object in dataframe automatically, therefore, the data should be converted back to string fromm obejct for further data handling. 

# In[12]:


#Rename the dataset columns name to lower case column name without space for better readability and naming conventions.
#https://github.com/melodyyip/NYC_accidents_2020/blob/main/NYC_accidents_GitHub.ipynb

mvc_crashes_vehicles_persons.rename(columns = {"CRASH DATE": "crash_date"}, inplace=True)
mvc_crashes_vehicles_persons.rename(columns = {"CRASH TIME": "crash_time"}, inplace=True)
mvc_crashes_vehicles_persons.rename(columns = {"BOROUGH": "borough"}, inplace=True)
mvc_crashes_vehicles_persons.rename(columns = {"ZIP CODE": "zip_code"}, inplace=True)
mvc_crashes_vehicles_persons.rename(columns = {"LATITUDE": "latitude"}, inplace=True)
mvc_crashes_vehicles_persons.rename(columns = {"LONGITUDE": "longitude"}, inplace=True)
mvc_crashes_vehicles_persons.rename(columns = {"LOCATION": "location"}, inplace=True)
mvc_crashes_vehicles_persons.rename(columns = {"ON STREET NAME": "on_street_name"}, inplace=True)
mvc_crashes_vehicles_persons.rename(columns = {"CROSS STREET NAME": "cross_street_name"}, inplace=True)
mvc_crashes_vehicles_persons.rename(columns = {"OFF STREET NAME": "off_street_name"}, inplace=True)
mvc_crashes_vehicles_persons.rename(columns = {"NUMBER OF PERSONS INJURED": "number_of_persons_injured"}, inplace=True)
mvc_crashes_vehicles_persons.rename(columns = {"NUMBER OF PERSONS KILLED": "number_of_persons_killed"}, inplace=True)
mvc_crashes_vehicles_persons.rename(columns = {"NUMBER OF PEDESTRIANS INJURED": "number_of_pedestrians_injured"}, inplace=True)
mvc_crashes_vehicles_persons.rename(columns = {"NUMBER OF PEDESTRIANS KILLED": "number_of_pedestrians_killed"}, inplace=True)
mvc_crashes_vehicles_persons.rename(columns = {"NUMBER OF CYCLIST INJURED": "number_of_cyclist_injured"}, inplace=True)
mvc_crashes_vehicles_persons.rename(columns = {"NUMBER OF CYCLIST KILLED": "number_of_cyclist_killed"}, inplace=True)
mvc_crashes_vehicles_persons.rename(columns = {"NUMBER OF MOTORIST INJURED": "number_of_motorist_injured"}, inplace=True)
mvc_crashes_vehicles_persons.rename(columns = {"NUMBER OF MOTORIST KILLED": "number_of_motorist_killed"}, inplace=True)
mvc_crashes_vehicles_persons.rename(columns = {"CONTRIBUTING FACTOR VEHICLE 1": "contributing_factor_vehicle_1"}, inplace=True)
mvc_crashes_vehicles_persons.rename(columns = {"CONTRIBUTING FACTOR VEHICLE 2": "contributing_factor_vehicle_2"}, inplace=True)
mvc_crashes_vehicles_persons.rename(columns = {"CONTRIBUTING FACTOR VEHICLE 3": "contributing_factor_vehicle_3"}, inplace=True)
mvc_crashes_vehicles_persons.rename(columns = {"CONTRIBUTING FACTOR VEHICLE 4": "contributing_factor_vehicle_4"}, inplace=True)
mvc_crashes_vehicles_persons.rename(columns = {"CONTRIBUTING FACTOR VEHICLE 5": "contributing_factor_vehicle_5"}, inplace=True)
mvc_crashes_vehicles_persons.rename(columns = {"COLLISION_ID": "collision_id"}, inplace=True)
mvc_crashes_vehicles_persons.rename(columns = {"VEHICLE TYPE CODE 1": "vehicle_type_code_1"}, inplace=True)
mvc_crashes_vehicles_persons.rename(columns = {"VEHICLE TYPE CODE 2": "vehicle_type_code_2"}, inplace=True)
mvc_crashes_vehicles_persons.rename(columns = {"VEHICLE TYPE CODE 3": "vehicle_type_code_3"}, inplace=True)
mvc_crashes_vehicles_persons.rename(columns = {"VEHICLE TYPE CODE 4": "vehicle_type_code_4"}, inplace=True)
mvc_crashes_vehicles_persons.rename(columns = {"VEHICLE TYPE CODE 5": "vehicle_type_code_5"}, inplace=True)


# Rename the above column name to lower case column without space for better readability and naming conventions.

# # Data cleaning is finished 

# # Exploratory Data Analysis

# In[13]:


#adding new column crash_year by extracting year part of the crash_date
mvc_crashes_vehicles_persons['crash_year'] = mvc_crashes_vehicles_persons['crash_date'].dt.year

mvc_crashes_vehicles_persons.head()


# In[14]:


#Getting unique years from crash_date
years = mvc_crashes_vehicles_persons['crash_date'].dt.year.unique()
years


# In[15]:


#creating master subset of the dataset for data analysis
mvc_crashes_vehicles_persons_subset = mvc_crashes_vehicles_persons[mvc_crashes_vehicles_persons['crash_year'].isin([2015, 2016, 2017, 2018, 2019, 2020])]


# In[16]:


# Get unique borough from the subset dataset
borough = mvc_crashes_vehicles_persons_subset['borough'].unique()
borough


# Because the naming convention is too long for coding, I am using a different naming convention for plotting the visualization.

# In[17]:


#short naming convention to plot visualisation.
RTI_df_subset_borough = mvc_crashes_vehicles_persons_subset[mvc_crashes_vehicles_persons_subset['borough'].isin(['MANHATTAN', 'BROOKLYN', 'BRONX', 'QUEENS', 'STATEN ISLAND'])]\



# In[18]:


RTI_df_subset_borough.info()


# In[19]:


#plotting a barchart to display maximum accident happen in the borough
borough= ['MANHATTAN', 'BROOKLYN', 'BRONX', 'QUEENS', 'STATEN ISLAND']

#plotting a barchart to display the maximum number of incidents happen in borough of New York City.
plt.figure(figsize=(12,6))
RTI_df_subset_borough.borough.hist(bins = 11,alpha=0.5,rwidth=0.90, color= ['red'])
plt.title('Number of Incidents in each borough', fontsize = 20)
plt.grid(False)
y_position = np.arange(len(borough))
plt.xticks(y_position ,borough)
plt.ylabel('Number of Incident' , fontsize = 16)
plt.xlabel('Borough', fontsize = 16,)


# The barchart displays total number incidents in each borough in the New york city during road traffic incidents in New York City in distinct year.

# In[20]:


#finding the outliers in the dataset by using the boxplot
sns.boxplot(x=mvc_crashes_vehicles_persons_subset["collision_id"])


# A box plot use boxes and lines to understand the distributions of one or more group of numeric data. The data distributin in the box plot shows that minimum collision happen is 3.2 and maximum is 3.6.

# Below PIE chart represent the contributing factors involved in road traffic incidents in New York City.

# In[21]:


#plotting a pie chart to display the Top 10 accident contributing factor percentage

RTI_df_subset_cf1=mvc_crashes_vehicles_persons_subset.contributing_factor_vehicle_1.value_counts(ascending=False).reset_index(name="count").head(10)

explode = (0.0, 0.0, 0.0 , 0.0 ,0.0,0.0, 0.0, 0.0 , 0.0 ,0.0) 
plt.figure(figsize=(10,9))
plt.pie(RTI_df_subset_cf1['count'],  labels=None, 
        autopct='%.1f',pctdistance=0.9, labeldistance=1.8 ,explode = explode, shadow=False, startangle=130,textprops={'fontsize': 14})
 
plt.axis('equal')
plt.legend(RTI_df_subset_cf1['index'], bbox_to_anchor=(1.2,0.8), loc="center right", fontsize=10, 
           bbox_transform=plt.gcf().transFigure)
plt.figtext(.5,.10,'Percentage of Top 10 Contributing Factor 1', fontsize=20, ha='center')
plt.show()


# In[22]:


#defining the correlation matrix for highly correlated variables
corr = mvc_crashes_vehicles_persons_subset.corr()
plt.subplots(figsize=(25,15))
sns.heatmap(corr)

A correlation heatmap is a heatmap that reperesnt a 2D correlation matrix between two distnict dimensions, the colour cells shows that data using colored cells to represent data from usually a monochromatic proportion. The values of the first dimension present as the rows of the table while of the second element as a column. 

Each square represent the correlation between the variables on both the axis. Correlation range are from +1 and -1. Values closer to zero means there is no linear pattern between the two variables. The close to 1 the correlation is the more positively correlated they are, when one increases so does the other and the closer to 1 the stronger this relationship is. Therefore, number of pedestrians killed is directly related to number of people killed on y- axis which clearly says that the variables are directly correlated.
# In[23]:


#plotting a pie chart to display the Top 10 accident contributing factor in percentage

RTI_df_subset_vtc1=mvc_crashes_vehicles_persons_subset.vehicle_type_code_4.value_counts(ascending=False).reset_index(name="count").head(10)

explode = (0.0, 0.0, 0.0 , 0.0 ,0.0,0.0, 0.0, 0.0 , 0.0 ,0.0) 
plt.figure(figsize=(10,9))
plt.pie(RTI_df_subset_vtc1['count'],  labels=None, 
        autopct='%.1f',pctdistance=0.9, labeldistance=1.8 ,explode = explode, shadow=False, startangle=130,textprops={'fontsize': 14})
 
plt.axis('equal')
plt.legend(RTI_df_subset_vtc1['index'], bbox_to_anchor=(1.2,0.8), loc="center right", fontsize=10, 
           bbox_transform=plt.gcf().transFigure)
plt.figtext(.5,.10,'Percentage of Top 10 Vehicle Type Involved', fontsize=20, ha='center')
plt.show()


# Th piechart represent the vehicle types involved in the road traffic incidents in new york city which clearly show that these vehicles like sport utility vehicle, sedan, 4 dr sedan are mostly involve in traffic incidents, followed by passenger vehicle.

# In[24]:


# Create weight column, using date month astype
heat_month_df = mvc_crashes_vehicles_persons_subset
heat_month_df['weight'] = heat_month_df['crash_date'].dt.month
heat_month_df['weight'] = heat_month_df['weight'].astype(float)

heat_month_df = heat_month_df.dropna(axis=0, subset=['latitude','longitude', 'weight'])

# List comprehension to make out list of lists
heat_month_data = [[[row['latitude'],row['longitude']] for index, row in heat_month_df[heat_month_df['weight'] == i].iterrows()] for i in range(0,12)]

map_month = folium.Map(location=[heat_month_df['latitude'].mean(), heat_month_df['longitude'].mean()],
                    zoom_start = 12) 


# Plot it on the map
hm = plugins.HeatMapWithTime(heat_month_data,auto_play=True,max_opacity=0.8)
hm.add_to(map_month)
# Display the map
map_month


# The geographical new york city heatmap originated from the shading matrices are used for highlighting and directly showing the matrix fields. The map shows the road traffic incidents took place in new york city with red colour on the basis of months.

# In[ ]:




