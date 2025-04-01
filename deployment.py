import pickle
import numpy as np
import streamlit as st

# Load the saved model from the file
filename = 'c:\\Users\\samal\\AVS CODE\\copy\\final_model.pkl'
with open(filename, 'rb') as file:
    loaded_model = pickle.load(file)

# Custom CSS for colorful representation
st.markdown(
    """
    <style>
    .title {
        color: #FF5733;
        text-align: center;
        font-size: 32px;
    }
    .text {
        color: #7D3C98;
        text-align: center;
        font-size: 18px;
    }
    .prediction {
        color: #6C3483;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create the Streamlit web app
st.markdown('<p class="title">Weight Prediction App</p>', unsafe_allow_html=True)
st.markdown('<p class="text">Enter your height in feet to predict your weight.</p>', unsafe_allow_html=True)

# Default value for height
default_height = 5.8

# Input height from the user
height_input = st.number_input("Enter the height in feet:", value=default_height, min_value=0.0)

# Predict button
if st.button('Predict'):
    # Reshape the input height to match the shape expected by the model (2D array)
    height_input_2d = np.array(height_input).reshape(1, -1)

    # Use the loaded model to make predictions
    predicted_weight = loaded_model.predict(height_input_2d)

    # Print the predicted weight
    st.markdown(f'<p class="prediction">Predicted weight: {predicted_weight[0, 0]} kg</p>', unsafe_allow_html=True)
