import streamlit as st
import requests
import numpy as np
import json


st.title("Cancer Histology Segmentation App")
uploaded_img = st.file_uploader("Choose Image")

if st.button("Run"):
    if uploaded_img is not None:
        # Send request to api
        url = "http://127.0.0.1:8000"
        payload = {}
        files = {"data": uploaded_img.getvalue()}
        headers = {}
        response = requests.post(url, files=files)
        # Read prediction
        prediction = json.loads(response.text)
        yhat = np.array(json.loads(prediction["Prediction"]))
        yhat = np.squeeze(np.where(yhat > 0.3, 1.0, 0.0))
        # Display input and output
        st.image(uploaded_img)
        for i in range(6):
            with st.container():
                for col in st.columns(6):
                    col.image(yhat[:, :, i])
