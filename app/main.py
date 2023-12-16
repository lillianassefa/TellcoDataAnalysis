# import streamlit as st
# import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler

# import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer
# from sklearn.cluster import KMeans

# sys.path.append('script')
# sys.path.append('analysis')


# from analysis import data_preprocessing as dbconn
# from analysis import user_engagement_analysis as user_engage
# from analysis import user_exprience_analysis as user_exp
# from analysis import user_overview_analysis as user_over




# def main():
#     st.title('Telecom Analysis Dashboard')

#     final_data= dbconn.data_preprocessing.clean.data()

#     engagment_metrics = user_engage.user_engagement_metrics(final_data)


#     st.subheader('Top 10 Engaged Customers')
#     st.dataframe(engagment_metrics.head(10))


#     st.subheader('Visualizing Engagement Metrics')
#     sns.pairplot(engagment_metrics)
#     st.pyplot()

# if __name__ == '__main__':
#     main() 
import streamlit as st
import sys
import os

# Add the project root to the system path
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
