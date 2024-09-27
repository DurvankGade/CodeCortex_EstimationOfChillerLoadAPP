import streamlit as st
import pandas as pd
import requests
from io import BytesIO

# Create a title for the app
st.title("Chiller Load Predictor")

# Create a file uploader
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

# Create a button to trigger the prediction
if st.button("Predict Chiller Load"):
    if uploaded_file is not None:
        # Send the file to the FastAPI endpoint
        response = requests.post("http://localhost:3000/predict", files={"csv_file": uploaded_file})

        # Check if the response was successful
        if response.status_code == 200:
            # Get the updated CSV data from the response
            updated_csv = response.content

            # Create a BytesIO object to hold the updated CSV data
            updated_csv_bytes = BytesIO(updated_csv)

            # Create a button to download the updated CSV file
            st.download_button("Download Updated CSV", updated_csv_bytes, file_name="updated_" + uploaded_file.name)

        else:
            st.error("Failed to predict chiller load")
    else:
        st.error("Please upload a CSV file")