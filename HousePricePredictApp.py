import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("D:\\PycharmProjects\\demo\\HousePricePrediction\\house_price_prediction_model_final.pkl")
cleaned_data = pd.read_csv("cleaned_data.csv")

# Define the input fields
input_fields = cleaned_data.columns.drop("SalePrice")

# Streamlit UI
st.title('House Price Prediction')

# Function to get user input
def get_user_input():
    user_input = {}
    for field in input_fields:
        user_input[field] = st.number_input(f"Enter {field}", step=1)
    return user_input

# Function to make prediction
def predict_price(user_input):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)
    return prediction[0]

# Main function
def main():
    user_input = get_user_input()
    if st.button('Predict'):
        prediction = predict_price(user_input)
        st.write(f"Predicted Price: {prediction}")

if __name__ == '__main__':
    main()
