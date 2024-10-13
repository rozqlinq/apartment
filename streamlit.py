import streamlit as st
from PIL import Image
import numpy as np
import torch
import pickle
from torchvision import transforms


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
    # image_tensor = preprocess_image(image)
    
    # Pass the image through the model to get predictions
    # with torch.no_grad():
    #     output = model(image_tensor)
    # num_rooms, total_area = output  # Unpack predictions (Example)

    # For this example, let's return dummy predictions
    num_rooms = np.random.randint(1, 5)  # Example: random number of rooms
    total_area = np.random.uniform(40, 120)  # Example: random total area in square meters
    return num_rooms, total_area

# Streamlit app UI
st.title("Apartment Floor Plan Analyzer")

st.write("Upload a floor plan image, and we'll predict the number of rooms and the total area.")

# Upload the image
uploaded_image = st.file_uploader("Upload an image of the floor plan", type=['png', 'jpg', 'jpeg'])

if uploaded_image:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Floor Plan", use_column_width=True)

    # Make a prediction
    if st.button('Analyze Floor Plan'):
        num_rooms, total_area = predict(image)
        st.write(f"Predicted Number of Rooms: {num_rooms}")
        st.write(f"Predicted Total Area: {total_area:.2f} square meters")
