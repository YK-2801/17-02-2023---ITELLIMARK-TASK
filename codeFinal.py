import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load datasets 
train_file = "train.csv" 
test_file = "test.csv"

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

# Handle missing values
train_df["Item_Weight"] = train_df.groupby("Item_Identifier")["Item_Weight"].transform(lambda x: x.fillna(x.mean()))
outlet_size_mode = train_df.groupby("Outlet_Type")["Outlet_Size"].apply(lambda x: x.mode().iloc[0])
train_df["Outlet_Size"] = train_df["Outlet_Size"].fillna(train_df["Outlet_Type"].map(outlet_size_mode))

# Standardize categorical values
train_df["Item_Fat_Content"] = train_df["Item_Fat_Content"].replace({"LF": "Low Fat", "low fat": "Low Fat", "reg": "Regular"})

# Encode categorical variables
encoder = LabelEncoder()

categorical_cols = ["Item_Fat_Content", "Outlet_Identifier", "Outlet_Size", "Outlet_Location_Type", "Outlet_Type", "Item_Type"]
for col in categorical_cols:
    train_df[col] = encoder.fit_transform(train_df[col])
train_df["Item_Identifier"] = encoder.fit_transform(train_df["Item_Identifier"])

# Define features and target
X = train_df.drop(columns=["Item_Outlet_Sales"])
y = train_df["Item_Outlet_Sales"]

# Split data for training and validation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = rf_model.predict(X_valid)
mae = mean_absolute_error(y_valid, y_pred)
rmse = mean_squared_error(y_valid, y_pred, squared=False)
r2 = r2_score(y_valid, y_pred)

print(f"MAE: {mae}, RMSE: {rmse}, R2 Score: {r2}")

# Prepare test dataset for predictions
test_df["Item_Weight"] = test_df.groupby("Item_Identifier")["Item_Weight"].transform(lambda x: x.fillna(x.mean()))
test_df["Outlet_Size"] = test_df["Outlet_Size"].fillna(test_df["Outlet_Type"].map(outlet_size_mode))
test_df["Item_Fat_Content"] = test_df["Item_Fat_Content"].replace({"LF": "Low Fat", "low fat": "Low Fat", "reg": "Regular"})
for col in categorical_cols:
    test_df[col] = encoder.fit_transform(test_df[col])
test_df["Item_Identifier"] = encoder.fit_transform(test_df["Item_Identifier"])

# Predict sales for test data
test_predictions = rf_model.predict(test_df.drop(columns=["Item_Outlet_Sales"], errors='ignore'))
print(test_predictions)

# Save predictions
test_df["Item_Outlet_Sales"] = test_predictions
acc_submission_format = test_df[["Item_Identifier","Outlet_Identifier","Item_Outlet_Sales"]]
acc_submission_format.to_excel("test_predictions.xlsx", index=False)
print("Predictions saved to test_predictions")