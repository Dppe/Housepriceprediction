import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
data = pd.read_csv("D:\PycharmProjects\demo\HousePricePrediction\HousePricePrediction.csv")
data = data.drop(columns=["Id"])
data = data.dropna()

# One-hot encode categorical variables
data = pd.get_dummies(data)

# Save the cleaned dataset
data.to_csv("cleaned_data.csv", index=False)

data = pd.read_csv("cleaned_data.csv")

# Split dataset into features and target variable
X = data.drop(columns=["SalePrice"])
y = data["SalePrice"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the training and testing sets
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
pd.DataFrame(y_train, columns=["SalePrice"]).to_csv("y_train.csv", index=False)
pd.DataFrame(y_test, columns=["SalePrice"]).to_csv("y_test.csv", index=False)

# Load training data
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")

# Initialize the Random Forest Regressor
model = RandomForestRegressor()
model.fit(X_train, y_train.values.ravel())
X_test = pd.read_csv("X_test.csv")

# Make predictions
predictions = model.predict(X_test)

# Save predictions to a file or use them for further analysis
pd.DataFrame(predictions, columns=["PredictedPrice"]).to_csv("predicted_prices.csv", index=False)

cleaned_data = pd.read_csv("cleaned_data.csv")
feature_columns = cleaned_data.columns.drop("SalePrice")

joblib.dump(model, "house_price_prediction_model_final.pkl")


