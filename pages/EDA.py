import os
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#-------- Title --------

st.title("EDA - Exploratory Data Analysis")
st.caption("Explore your dataset with various visualizations and insights.")

#-------- Get the CSV --------

if not os.path.exists("processed/df_clean.csv"):
    st.warning("Please complete Data Loading first.")
    st.stop()

df_clean = pd.read_csv("processed/df_clean.csv")

#-------- Get the session state --------

target_variable = st.session_state.get("target_variable") # get the target_variable from the "Data_Loading" page
target_features = st.session_state.get("target_features") # get the target_features from the "Data_Loading" page

#-------- Guards for the targets --------

if target_variable is None: 
    st.warning("Please select your Target Variable first.")
    st.stop()
    
if target_features is None: 
    st.warning("Please select your Target Features first.")
    st.stop()    
    
#-------- Histogram --------

fig = px.histogram(df_clean[target_variable], x=target_variable)
st.plotly_chart(fig)

#-------- Scatter Plot --------

fig = px.scatter(df_clean, x=target_variable, y=target_features[0])
st.plotly_chart(fig)

