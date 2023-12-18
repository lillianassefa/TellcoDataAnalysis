import matplotlib.pyplot as plt
import seaborn as sns
import sys
import streamlit as st
import sys
import os


sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from analysis import data_preprocessing as dbconn
from analysis import user_engagement_analysis as user_engage
from analysis import user_overview_analysis as user_over

def main():
    st.title('Telecom Analysis Dashboard')

    final_data = dbconn.read_data()
 
    engagement_metrics = user_engage.user_engagement_metrics(final_data)
    
    st.header('User Overview Analysis')
    st.subheader('Top Handsets Used by Customers')
    top_handset = user_over.top_handsets_identifier(final_data)
    st.dataframe(top_handset)

    st.subheader('Top 10 Engaged Customers')
    st.dataframe(engagement_metrics.head(10))

    st.subheader('Visualizing Engagement Metrics')
    fig, ax = plt.subplots()
    sns.pairplot(engagement_metrics)
    st.pyplot(fig)

if __name__ == '__main__':
    main()
