import pandas as pd 
import streamlit as st
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

#-------- Title --------

st.title("ML Pipeline")
st.caption("Build and evaluate your machine learning models with ease.")


#-------- Get the CSV from the session_state --------

df_clean = st.session_state.get("new_df_clean")
df_model = st.session_state.get("new_df_model")

if df_clean is None: # guard for the df_clean
    st.warning("Please complete Data Loading first.")
    st.stop()

if df_model is None: # guard for the df_model
    st.warning("Please complete Data Loading first.")
    st.stop()
    
#-------- Get targets from the session_state --------   
    
target_variable = st.session_state.get("saved_target_variable") # get the target_variable from the "Data_Loading" page
target_features = st.session_state.get("saved_target_features") # get the target_features from the "Data_Loading" page

#-------- Guards for the targets --------

if target_variable is None: 
    st.warning("Please select your Target Variable first.")
    st.stop()
    
if target_features is None: 
    st.warning("Please select your Target Features first.")
    st.stop()    
    
if target_variable not in df_model.columns:
    st.warning("Target variable not found in the model data. Please select a numeric target.")
    st.stop()

#----------------------------
#-------- Regression --------
#----------------------------

st.title("Regression Model Comparison")

x = df_model.drop(columns=target_variable) # drop the target variable from the features
y = df_model[target_variable]

st.subheader("Model Selection")
user_regression_model = st.selectbox("Select a Regression Model", options=["Linear Regression", "Random Forest"], key="reg_model")

#-------- User's Parameters --------

# for linear both models
test_size = st.slider("Test size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)

# defaults
user_n_estimators = 100
user_max_depth = 10
if user_regression_model == "Random Forest":
    # for random forest
    st.text("For Random Forest:")
    user_n_estimators = st.slider("Select how many trees you want:", min_value = 10, max_value = 200, value = 100, step = 10)
    user_max_depth = st.slider("Select the depth of the trees:", min_value = 2, max_value = 20, value = 10, step = 1)

# random_state=42 ensures reproducible results — same split every time the code runs
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

#-------- Linear Regression --------

lr_model = LinearRegression()
lr_model.fit(x_train,y_train)
lr_y_pred = lr_model.predict(x_test)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_y_pred))
lr_mae = mean_absolute_error(y_test, lr_y_pred)
lr_r2 = r2_score(y_test, lr_y_pred)


#-------- Random Forest --------

# oob_score=True enables out-of-bag evaluation — uses data not seen by each tree as a built-in validation method
rf_model = RandomForestRegressor(
    n_estimators = user_n_estimators,
    random_state=42,
    oob_score = True,
    max_depth=user_max_depth
    )

rf_model.fit(x_train, y_train)
rf_y_pred = rf_model.predict(x_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_y_pred))
rf_mae = mean_absolute_error(y_test, rf_y_pred)
rf_r2 = r2_score(y_test, rf_y_pred)


#-------- RMSE, R2 & Visualizations -------- 

if user_regression_model == "Linear Regression":
    st.subheader(" Linear Regression Measures")
    st.write("Root Mean Squared Error:", lr_rmse)
    st.write("Mean Absolute Error:", lr_mae)
    st.write("R2 score:", lr_r2)
    fig = px.scatter(x=y_test, y=lr_y_pred,
                    labels={'x': 'Actual', 'y':'Predicted'})
    st.plotly_chart(fig)
    
elif user_regression_model == "Random Forest":
    st.subheader("Random Forest Measures")
    st.write("Root Mean Squared Error:", rf_rmse)
    st.write("Mean Absolute Error:", rf_mae)
    st.write("R2 score:", rf_r2)
    fig = px.scatter(x=y_test, y=rf_y_pred,
                    labels={'x': 'Actual', 'y':'Predicted'})
    st.plotly_chart(fig)
    

#-------- Model Comparison --------    

st.subheader("Regression Model Comparison")
comparison = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'RMSE': [lr_rmse, rf_rmse],
    'MAE': [lr_mae, rf_mae],
    'R2': [lr_r2, rf_r2]
})
st.dataframe(comparison)

#----------------------------
#----- Classification -------
#----------------------------

st.divider()

st.title("Classification Model Comparison")

y_class = pd.qcut(y, q=3, labels=["Cheap", "Medium", "Expensive"]) # convert the regression target into 3 classes

st.subheader("Model Selection")
user_classification_model = st.selectbox("Select a Classification Model", options=["Decision Tree", "K-Nearest Neighbors"], key="class_model")


#-------- User's Parameters --------


# for both models
test_size = st.slider("Test size", min_value=0.1, max_value=0.5, value=0.2, step=0.05, key="class_test_size")

# defaults
user_max_depth_class = 10
user_n_neighbors = 5

if user_classification_model == "Decision Tree":
    # for decision tree
    st.text("For Decision Tree:")
    user_max_depth_class = st.slider("Select the depth of the tree:", min_value = 2, max_value = 20, value = 10, step = 1, key="max_depth_class")
    
elif user_classification_model == "K-Nearest Neighbors":
    # for KNN
    st.text("For K-Nearest Neighbors:")
    user_n_neighbors = st.slider("Select the number of neighbors:", min_value = 1, max_value = 20, value = 5, step = 1, key="n_neighbors")

x_train_class, x_test_class, y_train_class, y_test_class = train_test_split(x, y_class, test_size=test_size, random_state=42)
        
#-------- Decision Tree --------

dt_model = DecisionTreeClassifier(
    max_depth=user_max_depth_class,
    random_state=42
)
dt_model.fit(x_train_class, y_train_class)
dt_y_pred = dt_model.predict(x_test_class)
dt_accuracy = accuracy_score(y_test_class, dt_y_pred)
dt_f1 = f1_score(y_test_class, dt_y_pred, average='weighted')

#-------- K-Nearest Neighbors --------

knn_model = KNeighborsClassifier(n_neighbors=user_n_neighbors)
knn_model.fit(x_train_class, y_train_class)
knn_y_pred = knn_model.predict(x_test_class)
knn_accuracy = accuracy_score(y_test_class, knn_y_pred)
knn_f1 = f1_score(y_test_class, knn_y_pred, average='weighted')

#-------- Accuracy & Visualizations --------

if user_classification_model == "Decision Tree":
    st.subheader("Decision Tree Measures")
    st.write("Accuracy:", dt_accuracy)
    st.write("F1 Score:", dt_f1)
    cm = confusion_matrix(y_test_class, dt_y_pred)
    fig = px.imshow(cm, text_auto=True, 
                x=["Cheap", "Medium", "Expensive"],
                y=["Cheap", "Medium", "Expensive"],
                labels={'x': 'Predicted', 'y': 'Actual'})
    st.plotly_chart(fig)

elif user_classification_model == "K-Nearest Neighbors":
    st.subheader("K-Nearest Neighbors Measures")
    st.write("Accuracy:", knn_accuracy)
    st.write("F1 Score:", knn_f1)
    cm = confusion_matrix(y_test_class, knn_y_pred)
    fig = px.imshow(cm, text_auto=True, 
                x=["Cheap", "Medium", "Expensive"],
                y=["Cheap", "Medium", "Expensive"],
                labels={'x': 'Predicted', 'y': 'Actual'})
    st.plotly_chart(fig)

#-------- Model Comparison --------    

st.subheader("Classification Model Comparison")
comparison = pd.DataFrame({
    'Model': ['Decision Tree', 'K-Nearest Neighbors'],
    'Accuracy': [dt_accuracy, knn_accuracy],
    'F1 Score': [dt_f1, knn_f1]
})
st.dataframe(comparison)