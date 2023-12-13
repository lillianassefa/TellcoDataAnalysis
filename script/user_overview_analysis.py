import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def display_data_overview(df):
    st.title("Data Overview")
    st.text(f"Number of Rows: {df.shape[0]}")
    st.text(f"Number of Columns: {df.shape[1]}")
    st.text("Data Types:")
    st.write(df.dtypes)
    st.text("First Few Rows:")
    st.write(df.head())


def visualize_top_handsets(df):
    st.title("Top Handsets")
    top_handsets = top_handsets(df)
    st.bar_chart(top_handsets)

def visualize_top_manufacturers(df):
    st.title("Top Manufacturers")
    top_manufacturers = top_manufacturers(df)
    st.bar_chart(top_manufacturers)


def visualize_top_handsets_per_manufacturer(df):
    st.title("Top Handsets per Manufacturer")
    manufacturer_options = top_manufacturers(df).index.tolist()
    selected_manufacturer = st.selectbox("Select Manufacturer", manufacturer_options)
    top_handsets_manufacturer = top_handsets_per_manufacturer(df, selected_manufacturer)
    st.bar_chart(top_handsets_manufacturer)


def display_marketing_recommendation(df):
    st.title("Marketing Recommendation")
    recommendation = marketing_recommendation(df)
    st.text(recommendation)

