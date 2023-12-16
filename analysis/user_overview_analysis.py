import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

from analysis import data_preprocessing
from script import dbconn


#function used to histogram plot of the bearer Id count and the start time
def history_graph(df):
    sns.histplot(df['Bearer Id'], bins=30, kde=True)
    plt.show()

   # Distribution of Start times

    sns.histplot(df['Start'], bins=30, kde=True)
    plt.xticks(rotation=45)
    plt.show()

def missing_percent(df):
   tota1 = np.product(df.shape)
   missing = df.isnull().sum()

   totalmissing = missing.sum()
   percentage_missing =  round(((totalmissing/totalcells)*100), 2)
   return percentage_missing

#function used to list the top ten Handset Types used by customer

def top_handsets_identifier(df):
  
  top_handsets =  df['Handset Type'].value_counts().head(10)
#   print("The top 10 handset used by customers are: \n")
#   display(pd.DataFrame(top_handsets))
  return top_handsets  
#function used to define top handset manufacturers of Handets used by customers

def top_manfacturers(df):
   
   top_manfacturers = df['Handset Manufacturer'].value_counts().head(10)
   print("The top 10 manufacturers are: \n")
   display(pd.DataFrame(top_manfacturers).head)
   return top_manfacturers

#function used to identify the top 5 handsets per top 3 handset manufactureres

def handsets_per_manufacturers(df):
   top_manfacturers = df['Handset Manufacturer'].value_counts().head(10)
   top_manfacturers_names = top_manfacturers.index
   top_manfacturers_df = df[df['Handset Manufacturer'].isin(top_manfacturers_names)]

   top_handsets_per_manufacturer_df = top_manfacturers_df.groupby(['Handset Manufacturer','Handset Type']).size().reset_index(name='Count')
   print("Top 5 handsets per top 3 handset manufactureres:")
   display(pd.DataFrame(top_handsets_per_manufacturer_df))

# function used to calculate user aggregation information

def user_aggregation(df):
    
    applications = ['Social Media', 'Google', 'Email', 'Youtube', 'Netflix', 'Gaming', 'Other']

    user_aggregation = pd.DataFrame()
    grouped_by_user = df.groupby('MSISDN/Number')

    for app in applications:
       
    # Count the number of xDR sessions
       user_aggregation[f'{app}_Sessions'] = grouped_by_user['Bearer Id'].count()
    
    # Sum the session duration
       user_aggregation[f'{app}_Session_Duration'] = grouped_by_user['Dur. (ms)'].sum()
    
    # Sum the total download (DL) and upload (UL) data
       user_aggregation[f'{app}_Total_DL'] = grouped_by_user[f'{app} DL (Bytes)'].sum()
       user_aggregation[f'{app}_Total_UL'] = grouped_by_user[f'{app} UL (Bytes)'].sum()

    # Sum the total data volume (in Bytes) during this session for each application
       user_aggregation[f'{app}_Total_Volume'] = user_aggregation[f'{app}_Total_DL'] + user_aggregation[f'{app}_Total_UL']

# Reset the index to have 'MSISDN/Number' as a regular column
    user_aggregation.reset_index(inplace=True)

# Display the aggregated information
    display(pd.DataFrame(user_aggregation))

#function used to aggregate per user

def total_user_aggregation(df):
   df['Start'] = pd.to_datetime(df['Start'])
   user_aggregation = {
    'number_of_sessions': ('Bearer Id', 'count'),
    'session_duration': ('Dur. (ms)', 'sum'),
    'total_DL_data': ('Total DL (Bytes)', 'sum'),
    'total_UL_data': ('Total UL (Bytes)', 'sum'),
    }
   
   app_columns = ['Social Media', 'Google', 'Email', 'Youtube', 'Netflix', 'Gaming', 'Other']

   for app in app_columns:
      user_aggregation[f'{app}_data'] = (f'{app} DL (Bytes)', 'sum')
   user_data_aggregated = df.groupby('MSISDN/Number').agg(**user_aggregation).reset_index()
   return user_data_aggregated

#univariate analysis

#function to conduct dispersion parameters

def dispersion_parameters(df):
    dispersion_parameters = df.groupby('MSISDN/Number').agg({'Dur. (ms)': 'std', 'Total UL (Bytes)': 'std', 'Total DL (Bytes)': 'std'}).reset_index()
    dispersion_parameters.columns = ['MSISDN/Number', 'Duration_STD', 'Total_UL_STD', 'Total_DL_STD']
    return dispersion_parameters

#graphical univariate analysis

def histogram_of_session_duration(df):
    for column in ['Dur. (ms)', 'Total UL (Bytes)', 'Total DL (Bytes)']:
       
       plt.figure(figsize=(8, 6))
       sns.histplot(df[column], kde=True)
       plt.title(f'Histogram of {column}')
       plt.xlabel(column)
       plt.ylabel('Frequency')
       plt.show()


# Bivariate Analaysis

#Relationship between each application and the total DL+UL data

def bivariate_analysis(df):

    sampled_data = df.sample(frac=0.005) 
    sampled_data['Total Data (Bytes)'] = sampled_data['Total DL (Bytes)'] + sampled_data['Total UL (Bytes)']


    applications = ['Social Media', 'Google', 'Email', 'Youtube', 'Netflix', 'Gaming', 'Other']

    for app in applications:
       
       plt.figure(figsize=(10, 6))
       sns.scatterplot(data=sampled_data, x=f'{app} DL (Bytes)', y='Total Data (Bytes)')
       plt.title(f'{app} vs Total Data (DL+UL)')
       plt.xlabel(f'{app} Data (DL)')
       plt.ylabel('Total Data (DL+UL)')
       plt.show()

#function that shows correlation matrix of DL and UL of different sites

def correlation_matrix(df):

    correlation_matrix = df[['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)', 'Total UL (Bytes)', 'Total DL (Bytes)']].corr()
    display(correlation_matrix)


#function that shows the principal variance of the data
def principal_variance(df):
   features_for_pca = df[['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)', 'Total UL (Bytes)', 'Total DL (Bytes)']]
   imputer = SimpleImputer(strategy='mean')
   features_filled = imputer.fit_transform(features_for_pca)

# Standardize the data
   scaler = StandardScaler()
   features_scaled = scaler.fit_transform(features_filled)

# Apply PCA
   pca = PCA()
   principal_components = pca.fit_transform(features_scaled)

# Create a DataFrame with the principal components
   pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i}' for i in range(1, len(features_for_pca.columns) + 1)])

# Display the explained variance ratio
   explained_variance_ratio = pca.explained_variance_ratio_
   explained_variance_ratio

# Display principal components DataFrame
   display(pca_df)
   
   
   

