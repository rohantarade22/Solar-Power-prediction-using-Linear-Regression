import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


# Load sample dataset
data = pd.read_csv(r"C:\Users\Dell\Desktop\Solar power output prediction\dataset\solarpowergeneration.csv")
X=data.drop('generated_power_kw',axis=1)
y=data["generated_power_kw"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sidebar for model parameters
st.sidebar.header("Model Parameters")
n_estimators = st.sidebar.slider("Number of Trees (n_estimators)", 10, 200, 100)
max_depth = st.sidebar.slider("Max Depth", 2, 50, 10)

# Build the model
rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
rf_model.fit(X_train, y_train)

st.title("Solar Power Output Prediction")

# Show example features for user reference
st.write("Input features for the prediction :")

# User input fields
user_input = []
feature_names = X.columns
for feature in feature_names:
    value = st.number_input(f"Input value for {feature}",min_value=float(X[feature].min()),max_value=float(X[feature].max()), step=0.1)
    user_input.append(value)

# Prediction
if st.button("Predict"):
    user_input_array = np.array(user_input).reshape(1, -1)
    prediction = rf_model.predict(user_input_array)[0]
    st.subheader("Prediction Result")
    st.write(f"Predicted Outcome: {prediction:.2f}")

# Metrics evaluation on test data
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Model Performance on Test Data")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
st.write(f"R-squared (R²) Score: {r2:.2f}")
