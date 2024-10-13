import streamlit as st
from PIL import Image
import numpy as np
import torch
import pickle
from torchvision import transforms
from utils import apartment_type
from area import totalArea

model = pickle.load(open('apartment_room.pkl', 'rb'))

# Define image transformations if needed
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return preprocess(image).unsqueeze(0)  # Add batch dimension

# Define a prediction function (for example purposes, dummy predictions are used)
def predict(image):
    #image_tensor = preprocess_image(image)

    #Pass the image through the model to get predictions
    result = model(image, iou=0.5, save=True, project="x/", name="check", exist_ok=True)[0]
    #with torch.no_grad():
         #output = model(image_tensor)
    apartment, num_rooms = apartment_type(result)
    total_area = totalArea(image)
    
    return apartment, num_rooms, total_area

# Streamlit app UI
st.title("Apartment Floor Plan Analyzer")

st.write("Upload a floor plan image, and we'll predict the number of rooms and the total area.")

# Upload the image

uploaded_files = st.file_uploader("Upload images of floor plans", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        # Open the image
        image = Image.open(uploaded_file)

        # Display the uploaded image
        st.image(image, caption=f'Uploaded Floor Plan: {uploaded_file.name}', use_column_width=True)

        # Make prediction
        apartment, num_rooms, total_area = predict(image)

        # Display the results
        st.subheader(f'Results for {uploaded_file.name}:')
        st.write(f'Apartment Type: {apartment}')
        st.write(f'Number of Living Rooms: {num_rooms}')
        st.write(f'Total Area: {total_area} sq. units')

        st.write("---")  # Divider between results for different images
