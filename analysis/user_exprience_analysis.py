import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

def clean_data(df):
    # Replace missing values with the mean or mode
    df_cleaned = df.copy()
    df_cleaned['TCP Retransmission'] = df_cleaned['TCP Retransmission'].fillna(df_cleaned['TCP Retransmission'].mean())
    df_cleaned['RTT'] = df_cleaned['RTT'].fillna(df_cleaned['RTT'].mean())
    df_cleaned['Handset Type'] = df_cleaned['Handset Type'].fillna(df_cleaned['Handset Type'].mode()[0])
    df_cleaned['Throughput'] = df_cleaned['Throughput'].fillna(df_cleaned['Throughput'].mean())
    
    # Handling outliers if needed
    
    return df_cleaned

def aggregate_user_experience(df):
    # Aggregate per customer
    user_experience_agg = df.groupby('MSISDN/Number').agg({
        'TCP Retransmission': 'mean',
        'RTT': 'mean',
        'Handset Type': 'first',  # Assuming the handset type doesn't change for a user
        'Throughput': 'mean'
    }).reset_index()

    return user_experience_agg

def top_bottom_frequent_values(df, column, n=10):
    # Compute top, bottom, and most frequent values
    top_values = df[column].nlargest(n)
    bottom_values = df[column].nsmallest(n)
    frequent_values = df[column].value_counts().nlargest(n)

    return top_values, bottom_values, frequent_values

def compute_and_report_distribution(df, column_group, column_value, title):
    # Compute and report distribution
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=column_group, y=column_value, data=df)
    plt.title(title)
    plt.show()

def kmeans_clustering(df, features, n_clusters=3):
    # Perform K-means clustering
    scaler = StandardScaler()
    data_for_clustering = df[features].copy()
    data_for_clustering_scaled = scaler.fit_transform(data_for_clustering)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(data_for_clustering_scaled)

    # Provide a brief description of each cluster
    cluster_summary = df.groupby('Cluster')[features].mean()
    
    return df, cluster_summary

if __name__ == "__main__":
    # Load your dataset
    # df = pd.read_csv("your_dataset.csv")
    
    # Clean the data
    df_cleaned = clean_data(df)
    
    # Task 4.1 - Aggregate, per customer
    user_experience_agg = aggregate_user_experience(df_cleaned)
    print("Task 4.1 - Aggregated User Experience:")
    print(user_experience_agg.head())

    # Task 4.2 - Compute & list top, bottom, and most frequent values
    top_tcp, bottom_tcp, frequent_tcp = top_bottom_frequent_values(df_cleaned, 'TCP Retransmission')
    top_rtt, bottom_rtt, frequent_rtt = top_bottom_frequent_values(df_cleaned, 'RTT')
    top_throughput, bottom_throughput, frequent_throughput = top_bottom_frequent_values(df_cleaned, 'Throughput')

    print("\nTask 4.2 - Top, Bottom, and Most Frequent Values:")
    print("Top TCP Retransmission values:\n", top_tcp)
    print("\nBottom TCP Retransmission values:\n", bottom_tcp)
    print("\nMost Frequent TCP Retransmission values:\n", frequent_tcp)

    print("\nTop RTT values:\n", top_rtt)
    print("\nBottom RTT values:\n", bottom_rtt)
    print("\nMost Frequent RTT values:\n", frequent_rtt)

    print("\nTop Throughput values:\n", top_throughput)
    print("\nBottom Throughput values:\n", bottom_throughput)
    print("\nMost Frequent Throughput values:\n", frequent_throughput)

    # Task 4.3 - Compute & report distribution
    compute_and_report_distribution(user_experience_agg, 'Handset Type', 'Throughput', 'Task 4.3 - Distribution of Average Throughput per Handset Type')
    compute_and_report_distribution(user_experience_agg, 'Handset Type', 'TCP Retransmission', 'Task 4.3 - Average TCP Retransmission View per Handset Type')

    # Task 4.4 - K-means clustering
    features_for_clustering = ['TCP Retransmission', 'RTT', 'Throughput']
    df_clustered, cluster_summary = kmeans_clustering(user_experience_agg, features_for_clustering)
    print("\nTask 4.4 - K-means Clustering Results:")
    print("Clustered User Experience Data:")
    print(df_clustered.head())
    print("\nCluster Summary:")
    print(cluster_summary)
