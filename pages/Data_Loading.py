import os # for file path handling
import pandas as pd # for data manipulation
import streamlit as st # for web app development
from sklearn.preprocessing import MinMaxScaler, StandardScaler # for data normalization

#-------- Functions --------

# Function for Scaling (MinMax)
def apply_minmax(df):
    scaler = MinMaxScaler()
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'res_price']
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

# Function for Scaling (Standard(Z-Score))
def apply_zscore(df):
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'res_price']
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

# Function for the missing values mean + mode 
def fill_missing_mean(df):
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())
    categorical_cols = df.select_dtypes(exclude='number').columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df

# Function for the missing values using median + mode
def fill_missing_median(df):
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    categorical_cols = df.select_dtypes(exclude='number').columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df

#-------- Title --------

st.title("Data Loading & Preprocessing")
st.caption("Upload your dataset, clean it, and prepare it for analysis.")

#-------- Data Reading --------

# File Uploading
uploaded_file = st.file_uploader('Upload your csv here.', type=["csv"], help=None)
if uploaded_file is None:
    st.info('Please upload a file to continue.')
    st.stop()
    
# Reading the csv
dataset = pd.read_csv(uploaded_file)

# Displaying the Dataset
st.dataframe(dataset.head())
st.write(f"Dataset shape: {dataset.shape[0]} rows and {dataset.shape[1]} columns")
st.write(f"Missing values per column:")
st.dataframe(dataset.isnull().sum())
st.dataframe(dataset.dtypes)

# Columns for options
col1, col2, col3 = st.columns(3)

with col1: 
    remove_duplicates = st.checkbox(
            "Remove duplicate rows",
            value=True
    )

with col2: 
    missing_strategy = st.selectbox(
            "Missing-value handling",
            [
                "None",
                "Drop rows with NaN",
                "Fill numeric with mean + categorical with mode",
                "Fill numeric with median + categorical with mode"
            ]
    )
       
with col3:
    scaling_strategy = st.selectbox(
        "Scaling for numeric features",
        ["None", "MinMax", "Standard(Z-Score)"]
    )

# Cleaning Data
df_clean = dataset.copy() 
df_clean = df_clean[df_clean['deleted'] == 0] # Keeping only active listings (deleted=0)
df_clean = df_clean[(df_clean['res_price'] >= 30000) & (df_clean['res_price'] < 2670000)] # Deleting the rows where the price isn't between 30.000 and 2.670.000
freq = df_clean['res_address'].value_counts() # Counting how many times each address appears so.. 
df_clean['neighborhood'] = df_clean['res_address'].where( #..if it appears 10 or more i keep the name..
    df_clean['res_address'].isin(freq[freq >= 10].index), other='Other' #..if it appears less i replace it with "Other"
)
df_clean = df_clean.drop(columns=['parking', 'res_price_sqr', 'deleted', 'deleted_at', 'res_address', 'location_name', 'res_date']) # Dropping the columns i won't need


#-------- Missing Valuse Handling --------

# Cases for the missing_strategy
if missing_strategy == "None": # if the users wants no changes just move on
    pass
elif missing_strategy == "Drop rows with NaN": # if the user want to drop the rows with NaN
    df_clean = df_clean.dropna()               # drop them 
elif missing_strategy == "Fill numeric with mean + categorical with mode":
    df_clean = fill_missing_mean(df_clean)     # fill_missing_mean function
elif missing_strategy == "Fill numeric with median + categorical with mode":
    df_clean = fill_missing_median(df_clean)   # fill_missing_median function 

#-------- Duplicate Values Handling --------

# Remove Duplicates
if remove_duplicates:
    df_clean = df_clean.drop_duplicates()


df_model = df_clean.copy() # we want to keep our original clean dataset for eda, so for encoding we create a copy

#-------- Scaling Strategy --------
    
# Cases for the scaling_strategy
if scaling_strategy == "None": # if the users wants no changes just move on
    pass
elif scaling_strategy == "MinMax":
    df_model = apply_minmax(df_model)
elif scaling_strategy == "Standard(Z-Score)":
    df_model = apply_zscore(df_model)

#-------- Encoding --------

categorical_cols = df_model.select_dtypes(include=['object']).columns.tolist()
df_model = pd.get_dummies(df_model, columns=categorical_cols)

# Saving the data
os.makedirs("processed", exist_ok=True)
df_model.to_csv("processed/df_model.csv", index=False)
df_clean.to_csv("processed/df_clean.csv", index=False)

# Displaying the clean Dataset
st.subheader("Cleaned Data Preview")
st.dataframe(df_clean.head())
st.write(f"Dataset shape: {df_clean.shape[0]} rows and {df_clean.shape[1]} columns")
st.success(f"Your data has been saved! You can proceed to EDA.")