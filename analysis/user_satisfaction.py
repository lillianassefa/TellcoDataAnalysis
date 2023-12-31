import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import joblib
import pymysql

def assign_engagement_experience_scores(df, clustering_results, features_for_clustering):
  
    scaler = StandardScaler()
    user_data_for_clustering = df[features_for_clustering].copy()
    user_data_for_clustering_scaled = scaler.fit_transform(user_data_for_clustering)

    less_engaged_cluster_center = clustering_results.groupby('Cluster')[features_for_clustering].mean().iloc[0]
    engagement_scores = np.linalg.norm(user_data_for_clustering_scaled - less_engaged_cluster_center, axis=1)


    worst_experience_cluster_center = clustering_results.groupby('Cluster')[features_for_clustering].mean().idxmax()
    experience_scores = np.linalg.norm(user_data_for_clustering_scaled - worst_experience_cluster_center, axis=1)

    df['Engagement Score'] = engagement_scores
    df['Experience Score'] = experience_scores

    return df

def calculate_satisfaction_score(df):
 
    df['Satisfaction Score'] = (df['Engagement Score'] + df['Experience Score']) / 2
    return df

def top_satisfied_customers(df, n=10):
 
    top_satisfied = df.nlargest(n, 'Satisfaction Score')
    return top_satisfied

def build_regression_model(df, target_column='Satisfaction Score'):
    
    features = ['Engagement Score', 'Experience Score']
    X = df[features]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

   
    y_pred = model.predict(X_test)

   
    mse = mean_squared_error(y_test, y_pred)

    return model, mse

def kmeans_clustering_2(df, features, n_clusters=2):
    
    scaler = StandardScaler()
    data_for_clustering = df[features].copy()
    data_for_clustering_scaled = scaler.fit_transform(data_for_clustering)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster 2'] = kmeans.fit_predict(data_for_clustering_scaled)

   
    cluster_summary = df.groupby('Cluster 2')[features].mean()

    return df, cluster_summary

def aggregate_scores_per_cluster(df, cluster_column='Cluster 2'):
   
    agg_per_cluster = df.groupby(cluster_column)['Satisfaction Score', 'Experience Score'].mean()
    return agg_per_cluster

def export_to_mysql(df, table_name='satisfaction_scores', host='localhost', user='root', password='password', database='telecom_analysis'):
   
    connection = pymysql.connect(host=host, user=user, password=password, database=database)
    cursor = connection.cursor()

    
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        `MSISDN/Number` INT PRIMARY KEY,
        `Engagement Score` FLOAT,
        `Experience Score` FLOAT,
        `Satisfaction Score` FLOAT,
        `Cluster 2` INT
    );
    """
    cursor.execute(create_table_query)

   
    insert_query = f"""
    INSERT INTO {table_name} (`MSISDN/Number`, `Engagement Score`, `Experience Score`, `Satisfaction Score`, `Cluster 2`)
    VALUES (%s, %s, %s, %s, %s);
    """

    for _, row in df.iterrows():
        cursor.execute(insert_query, (row['MSISDN/Number'], row['Engagement Score'], row['Experience Score'], row['Satisfaction Score'], row['Cluster 2']))

    connection.commit()
    connection.close()

if __name__ == "__main__":
    
    features_for_clustering = ['TCP Retransmission', 'RTT', 'Throughput']

    df = assign_engagement_experience_scores(df, clustering_results, features_for_clustering)
   
    print(df[['MSISDN/Number', 'Engagement Score', 'Experience Score']].head())

   
    df = calculate_satisfaction_score(df)
    top_satisfied = top_satisfied_customers(df, n=10)
    print("\nTask 5.2 - Top Satisfied Customers:")
    print(top_satisfied[['MSISDN/Number', 'Satisfaction Score']])

   
    regression_model, mse = build_regression_model(df)
    print("\nTask 5.3 - Regression Model Evaluation:")
    print(f"Mean Squared Error: {mse}")


