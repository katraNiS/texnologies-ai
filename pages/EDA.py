import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#-------- Title --------

st.title("EDA - Exploratory Data Analysis")
st.caption("Explore your dataset with various visualizations and insights.")

#-------- Get the CSV from the session_state --------

df_clean = st.session_state.get("new_df_clean")
df_model = st.session_state.get("new_df_model")

if df_clean is None: # guard for the df_clean
    st.warning("Please complete Data Loading first.")
    st.stop()

if df_model is None: # guard for the df_model
    st.warning("Please complete Data Loading first.")
    st.stop()
    
#-------- Get the targets from session_state --------

target_variable = st.session_state.get("saved_target_variable") # get the target_variable from the "Data_Loading" page
target_features = st.session_state.get("saved_target_features") # get the target_features from the "Data_Loading" page

#-------- Guards for the targets --------

if target_variable is None: 
    st.warning("Please select your Target Variable first.")
    st.stop()
    
if target_features is None: 
    st.warning("Please select your Target Features first.")
    st.stop()    
    
#-------- Histogram --------

st.subheader("Histogram")

prev_variable = st.selectbox("Select a Variable to Preview", 
    options=df_clean.columns.tolist(), key="hist_variable")

fig = px.histogram(df_clean[prev_variable], x=prev_variable)
st.plotly_chart(fig)

#-------- Box Plot --------

st.subheader("Box Plot")
st.caption("The box plot shows the same variable as the histogram above.")

fig = px.box(df_clean[prev_variable], x= prev_variable)
st.plotly_chart(fig)

#-------- Scatter Plot --------

st.subheader("Scatter Plot")

scatter_variable = st.selectbox("Select a Variable to Preview", 
    options=df_clean.columns.tolist(), key="scatter_variable")
scatter_features = st.selectbox("Select a Feature to Preview",
    options=df_clean.columns.tolist(), key="scatter_features")

fig = px.scatter(df_clean, x=scatter_variable, y=scatter_features)
st.plotly_chart(fig)

#-------- Correlation Heatmap --------

st.subheader("Correlation Heatmap")
correlation_matrix = df_clean.select_dtypes(include='number').corr()
fig = px.imshow(correlation_matrix, x=correlation_matrix.columns, y=correlation_matrix.columns, text_auto=True, aspect="auto")
st.plotly_chart(fig)

#-------- PCA --------

st.subheader("PCA")
categorical_cols = df_clean.select_dtypes(include='object').columns.tolist()
pca_color = st.selectbox("Color PCA by", options=categorical_cols)

# Dropping NaN values because PCA cannot handle missing data
numeric_data = df_clean.select_dtypes(include="number").dropna()

# StandardScaler before PCA — necessary because PCA is sensitive to feature scale. Without scaling, features with larger values (e.g. res_price) would dominate the analysis
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

fig = px.scatter(x=pca_result[:,0], y=pca_result[:,1], color=df_clean.loc[numeric_data.index, pca_color], labels={'x': 'PC1', 'y': 'PC2'})
# Using numeric_data.index to match rows — after dropna() some rows may have been removed, so we need the same rows from df_clean for coloring
st.plotly_chart(fig)

st.dataframe(pd.DataFrame(pca.components_, columns=numeric_data.columns, index=['PC1', 'PC2']))

if st.button(f"Press to continue to proceed to the ML Pipeline."):
    st.switch_page("pages/ML_Pipeline.py")