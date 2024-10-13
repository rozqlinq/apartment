from paddleocr import PaddleOCR
import cv2
import re
from PIL import Image
import numpy as np
import pandas as pd

# Initialize PaddleOCR
    
def totalArea(image):
    ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Load the OCR model
    
    cimage = np.array(image)
    
    gray = cv2.cvtColor(cimage, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise and improve text clarity
    #blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding to make text stand out
    _, binary_image = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    # Resize the image for better OCR accuracy (if the image is too small)
    height, width = binary_image.shape
    if height < 600 or width < 600:
        binary_image = cv2.resize(binary_image, (width * 2, height * 2), interpolation=cv2.INTER_LINEAR)
    
    # Perform OCR with PaddleOCR on the preprocessed image
    results = ocr.ocr(binary_image, cls=True)
    
    # Extract decimal numbers using a regex pattern
    decimal_numbers = []
    for line in results:
        for res in line:
            text = res[1][0]  # Extract detected text
            # Updated regex to match only decimal numbers
            text = text.replace(',','.')
            matches = re.findall(r'\b\d+\.\d+\b', text)  # Matches numbers with at least one decimal point
            decimal_numbers.extend(matches)
    
    # Clean up the extracted numbers (if necessary)
    cleaned_numbers = []
    for number in decimal_numbers:
        cleaned = number.strip()  # Strip any whitespace
        cleaned_numbers.append(cleaned)
    
    # Print or save the cleaned decimal numbers
    float_numbers = [float(num) for num in cleaned_numbers if num]  # Convert to float, skipping empty strings
    total_sum = sum(float_numbers)
    
    rounded_sum = round(total_sum, 2)
    
    return rounded_sum
