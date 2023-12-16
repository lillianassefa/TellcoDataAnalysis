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
    # Assign engagement and experience scores to each user
    scaler = StandardScaler()
    user_data_for_clustering = df[features_for_clustering].copy()
    user_data_for_clustering_scaled = scaler.fit_transform(user_data_for_clustering)

    # Assign engagement score
    less_engaged_cluster_center = clustering_results.groupby('Cluster')[features_for_clustering].mean().iloc[0]
    engagement_scores = np.linalg.norm(user_data_for_clustering_scaled - less_engaged_cluster_center, axis=1)

    # Assign experience score
    worst_experience_cluster_center = clustering_results.groupby('Cluster')[features_for_clustering].mean().idxmax()
    experience_scores = np.linalg.norm(user_data_for_clustering_scaled - worst_experience_cluster_center, axis=1)

    df['Engagement Score'] = engagement_scores
    df['Experience Score'] = experience_scores

    return df

def calculate_satisfaction_score(df):
    # Calculate satisfaction score as the average of engagement and experience scores
    df['Satisfaction Score'] = (df['Engagement Score'] + df['Experience Score']) / 2
    return df

def top_satisfied_customers(df, n=10):
    # Report the top satisfied customers
    top_satisfied = df.nlargest(n, 'Satisfaction Score')
    return top_satisfied

def build_regression_model(df, target_column='Satisfaction Score'):
    # Build a regression model to predict satisfaction score
    features = ['Engagement Score', 'Experience Score']
    X = df[features]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)

    return model, mse

def kmeans_clustering_2(df, features, n_clusters=2):
    # Perform K-means clustering
    scaler = StandardScaler()
    data_for_clustering = df[features].copy()
    data_for_clustering_scaled = scaler.fit_transform(data_for_clustering)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster 2'] = kmeans.fit_predict(data_for_clustering_scaled)

    # Provide a brief description of each cluster
    cluster_summary = df.groupby('Cluster 2')[features].mean()

    return df, cluster_summary

def aggregate_scores_per_cluster(df, cluster_column='Cluster 2'):
    # Aggregate average satisfaction and experience scores per cluster
    agg_per_cluster = df.groupby(cluster_column)['Satisfaction Score', 'Experience Score'].mean()
    return agg_per_cluster

def export_to_mysql(df, table_name='satisfaction_scores', host='localhost', user='root', password='password', database='telecom_analysis'):
    # Export the final table to a local MySQL database
    connection = pymysql.connect(host=host, user=user, password=password, database=database)
    cursor = connection.cursor()

    # Create the table if it doesn't exist
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

    # Insert data into the table
    insert_query = f"""
    INSERT INTO {table_name} (`MSISDN/Number`, `Engagement Score`, `Experience Score`, `Satisfaction Score`, `Cluster 2`)
    VALUES (%s, %s, %s, %s, %s);
    """

    for _, row in df.iterrows():
        cursor.execute(insert_query, (row['MSISDN/Number'], row['Engagement Score'], row['Experience Score'], row['Satisfaction Score'], row['Cluster 2']))

    connection.commit()
    connection.close()

if __name__ == "__main__":
    # Uncomment and replace the following line with the actual path to your dataset
    # df = pd.read_csv("path/to/your/dataset.csv")

    # Load clustering results from Task 4.4
    # clustering_results = pd.read_csv("clustering_results.csv")

    # Uncomment and replace the following line with the actual path to your dataset used for experience analysis
    # df_experience = pd.read_csv("path/to/your/experience_dataset.csv")

    # Features for clustering
    features_for_clustering = ['TCP Retransmission', 'RTT', 'Throughput']

    # Task 5.1 - Assign engagement and experience scores
    df = assign_engagement_experience_scores(df, clustering_results, features_for_clustering)
    print("Task 5.1 - Assigned Engagement and Experience Scores:")
    print(df[['MSISDN/Number', 'Engagement Score', 'Experience Score']].head())

    # Task 5.2 - Calculate satisfaction score and report top satisfied customers
    df = calculate_satisfaction_score(df)
    top_satisfied = top_satisfied_customers(df, n=10)
    print("\nTask 5.2 - Top Satisfied Customers:")
    print(top_satisfied[['MSISDN/Number', 'Satisfaction Score']])

    # Task 5.3 - Build a regression model
    regression_model, mse = build_regression_model(df)
    print("\nTask 5.3 - Regression Model Evaluation:")
    print(f"Mean Squared Error: {mse}")

    # Task 5.4 - Run
