import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

def clean_data(df):
    
    df_cleaned = df.copy()
    df_cleaned['TCP Retransmission'] = df_cleaned['TCP Retransmission'].fillna(df_cleaned['TCP Retransmission'].mean())
    df_cleaned['RTT'] = df_cleaned['RTT'].fillna(df_cleaned['RTT'].mean())
    df_cleaned['Handset Type'] = df_cleaned['Handset Type'].fillna(df_cleaned['Handset Type'].mode()[0])
    df_cleaned['Throughput'] = df_cleaned['Throughput'].fillna(df_cleaned['Throughput'].mean())
    
   
    return df_cleaned

def aggregate_user_experience(df):
   
    user_experience_agg = df.groupby('MSISDN/Number').agg({
        'TCP Retransmission': 'mean',
        'RTT': 'mean',
        'Handset Type': 'first',  # Assuming the handset type doesn't change for a user
        'Throughput': 'mean'
    }).reset_index()

    return user_experience_agg

def top_bottom_frequent_values(df, column, n=10):
   
    top_values = df[column].nlargest(n)
    bottom_values = df[column].nsmallest(n)
    frequent_values = df[column].value_counts().nlargest(n)

    return top_values, bottom_values, frequent_values

def compute_and_report_distribution(df, column_group, column_value, title):
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=column_group, y=column_value, data=df)
    plt.title(title)
    plt.show()

def kmeans_clustering(df, features, n_clusters=3):
    
    scaler = StandardScaler()
    data_for_clustering = df[features].copy()
    data_for_clustering_scaled = scaler.fit_transform(data_for_clustering)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(data_for_clustering_scaled)

   
    cluster_summary = df.groupby('Cluster')[features].mean()
    
    return df, cluster_summary


