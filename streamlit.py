import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
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
    result = model(image, iou=0.5, name="check", exist_ok=True)[0]
    #with torch.no_grad():
         #output = model(image_tensor)
    apartment, num_rooms = apartment_type(result)
    total_area = totalArea(image)
    
    return apartment, num_rooms, total_area

# Streamlit app UI
st.set_page_config(layout='centered')
st.image('Header.jpg', width=700)

st.markdown("""
    <style>
    .stApp {
        background-color: #f5f5f5;
    }
    .title {
        font-family: 'Verdana';
        color: #ff6347;
    }
    .header {
        font-family: 'Verdana';
        color: #4682b4;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Автоматическое определение количества жилых комнат и общей площади квартиры")

st.write("Загрузите изображения планировок квартир, чтобы определить количество жилых комнат и общей площади:")

# Upload the image

uploaded_files = st.file_uploader("Загрузите изображения планировок квартир ниже", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
results = []

if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        # Open the image
        image = Image.open(uploaded_file)

        # Display the uploaded image
        st.image(image, caption=f'Загруженная планировка: {uploaded_file.name}', use_column_width=True)

        # Make prediction
        apartment, num_rooms, total_area = predict(image)

        results.append({
            "File Name": uploaded_file.name,
            "Apartment Type": apartment,
            "Number of Rooms": num_rooms,
            "Total Area (sq. units)": total_area
        })

        # Display the results
        st.subheader(f'Результаты для {uploaded_file.name}:')
        st.write(f'Тип квартиры: {apartment}')
        st.write(f'Количество жилых комнат: {num_rooms}')
        st.write(f'Общая площадь: {total_area} sq. units')

        st.write("---")  # Divider between results for different images
        
    if results:
        df = pd.DataFrame(results)
        
        # Display the table
        st.subheader("Результаты предсказаний")
        st.dataframe(df)
