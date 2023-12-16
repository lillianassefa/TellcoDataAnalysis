import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer 
from IPython.display import display



# funtions used to find the user engagement metrics per customer

def user_engagement_metrics(df):
    engagement_metrics = df.groupby('MSISDN/Number').agg({
        'Dur. (ms)': 'sum',  # Duration of the session
        'Total UL (Bytes)': 'sum',  # Total upload traffic
        'Total DL (Bytes)': 'sum',  # Total download traffic
        'Bearer Id': 'count'  # Sessions frequency
    }).reset_index()


    engagement_metrics.columns = ['MSISDN/Number', 'Total Session Duration', 'Total UL Traffic', 'Total DL Traffic', 'Session Frequency']

    # Display the top 10 customers per engagement metric
    top_10_duration = engagement_metrics.sort_values('Total Session Duration', ascending=False).head(5)
    top_10_ul_traffic = engagement_metrics.sort_values('Total UL Traffic', ascending=False).head(5)
    top_10_dl_traffic = engagement_metrics.sort_values('Total DL Traffic', ascending=False).head(5)
    top_10_session_frequency = engagement_metrics.sort_values('Session Frequency', ascending=False).head(5)

    display(top_10_duration, top_10_ul_traffic, top_10_dl_traffic, top_10_session_frequency)
    
    return engagement_metrics

# function used find the normalized engagment metrics

def normalized_engagment_metrics(df):
    engagement_metrics = df.groupby('MSISDN/Number').agg({
        'Dur. (ms)': 'sum',  # Duration of the session
        'Total UL (Bytes)': 'sum',  # Total upload traffic
        'Total DL (Bytes)': 'sum',  # Total download traffic
        'Bearer Id': 'count'  # Sessions frequency
    }).reset_index()
   
    engagement_metrics.columns = ['MSISDN/Number', 'Total Session Duration', 'Total UL Traffic', 'Total DL Traffic', 'Session Frequency']
    scaler = StandardScaler()
    normalized_engagement_metrics = scaler.fit_transform(engagement_metrics[['Total Session Duration', 'Total UL Traffic', 'Total DL Traffic', 'Session Frequency']])


    # Run k-means clustering

    kmeans = KMeans(n_clusters=3, random_state=42)
    engagement_metrics['Cluster'] = kmeans.fit_predict(normalized_engagement_metrics)
       # Display minimum, maximum, average, and total non-normalized metrics for each cluster
    cluster_summary = engagement_metrics.groupby('Cluster').agg({
        'Total Session Duration': ['min', 'max', 'mean', 'sum'],
        'Total UL Traffic': ['min', 'max', 'mean', 'sum'],
        'Total DL Traffic': ['min', 'max', 'mean', 'sum'],
        'Session Frequency': ['min', 'max', 'mean', 'sum']
    }).reset_index()

    return cluster_summary


# aggregated app traffic for the user usage per application

def app_traffic(df):
    app_traffic = df.groupby('MSISDN/Number').agg({
        'Social Media DL (Bytes)': 'sum',
        'Google DL (Bytes)': 'sum',
        'Email DL (Bytes)': 'sum',
        'Youtube DL (Bytes)': 'sum',
        'Netflix DL (Bytes)': 'sum',
        'Gaming DL (Bytes)': 'sum',
        'Other DL (Bytes)': 'sum',
        'Social Media UL (Bytes)': 'sum',
        'Google UL (Bytes)': 'sum',
        'Email UL (Bytes)': 'sum',
        'Youtube UL (Bytes)': 'sum',
        'Netflix UL (Bytes)': 'sum',
        'Gaming UL (Bytes)': 'sum',
        'Other UL (Bytes)': 'sum'
    }).reset_index()

    # Correct column names for aggregation
    app_traffic.columns = ['MSISDN/Number', 'Social Media', 'Google', 'Email', 'Youtube', 'Netflix', 'Gaming', 'Other',
                           'Social Media UL', 'Google UL', 'Email UL', 'Youtube UL', 'Netflix UL', 'Gaming UL', 'Other UL']

    # Display the top 10 most engaged users per application
    top_10_social_media = app_traffic.sort_values(['Social Media', 'Social Media UL'], ascending=False).head(10)
    top_10_google = app_traffic.sort_values(['Google', 'Google UL'], ascending=False).head(10)
    top_10_youtube = app_traffic.sort_values(['Youtube', 'Youtube UL'], ascending=False).head(10)

    # Display the top 10 most engaged users per application
    display(top_10_social_media, top_10_google, top_10_youtube)

    # Plot pairplot for engagement metrics
    engagement_metrics = df[['Total Session Duration', 'Total UL Traffic', 'Total DL Traffic', 'Session Frequency']]
    sns.pairplot(engagement_metrics)
    plt.suptitle('Scatter Plot Matrix of Engagement Metrics', y=1.02)
    plt.show()


# Top Applications used by users

def top_apps_per_user_engagement(df):
    app_traffic = df.groupby('MSISDN/Number').agg({
        'Social Media DL (Bytes)': 'sum',
        'Google DL (Bytes)': 'sum',
        'Email DL (Bytes)': 'sum',
        'Youtube DL (Bytes)': 'sum',
        'Netflix DL (Bytes)': 'sum',
        'Gaming DL (Bytes)': 'sum',
        'Other DL (Bytes)': 'sum',
        'Social Media UL (Bytes)': 'sum',
        'Google UL (Bytes)': 'sum',
        'Email UL (Bytes)': 'sum',
        'Youtube UL (Bytes)': 'sum',
        'Netflix UL (Bytes)': 'sum',
        'Gaming UL (Bytes)': 'sum',
        'Other UL (Bytes)': 'sum'
    }).reset_index()

    top_3_apps = app_traffic.sum().sort_values(ascending=False).head(4)
    plt.figure(figsize=(10, 6))
    top_3_apps[1:].plot(kind='bar', color='skyblue')  # Exclude the first element (MSISDN/Number)
    plt.title('Top 3 Most Used Applications')
    plt.xlabel('Application')
    plt.ylabel('Total Traffic (Bytes)')
    plt.show()


def KElbowVisualizer_on_normalized_metrics(df):
    engagement_metrics = df.groupby('MSISDN/Number').agg({
        'Dur. (ms)': 'sum',  # Duration of the session
        'Total UL (Bytes)': 'sum',  # Total upload traffic
        'Total DL (Bytes)': 'sum',  # Total download traffic
        'Bearer Id': 'count'  # Sessions frequency
    }).reset_index()
    engagement_metrics.columns = ['MSISDN/Number', 'Total Session Duration', 'Total UL Traffic', 'Total DL Traffic', 'Session Frequency']
    scaler = StandardScaler()
    normalized_engagement_metrics = scaler.fit_transform(engagement_metrics[['Total Session Duration', 'Total UL Traffic', 'Total DL Traffic', 'Session Frequency']])
    # Fit the KElbowVisualizer on the normalized engagement metrics data
    visualizer = KElbowVisualizer(KMeans(), k=(4), metric='distortion')
    visualizer.fit(normalized_engagement_metrics)
    visualizer.show()



