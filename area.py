from paddleocr import PaddleOCR
import cv2
import re
import numpy as np
import pandas as pd

# Initialize PaddleOCR
    
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def totalArea(image):
    # Convert PIL image to a NumPy array (RGB to BGR for OpenCV)
    cimage = np.array(image)
    
    # Convert to grayscale for better OCR accuracy
    gray = cv2.cvtColor(cimage, cv2.COLOR_RGB2GRAY)
    
    # Apply adaptive thresholding to improve text visibility
    _, binary_image = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    # Resize only if the image dimensions are smaller than a threshold
    height, width = binary_image.shape
    if height < 600 or width < 600:
        binary_image = cv2.resize(binary_image, (width * 2, height * 2), interpolation=cv2.INTER_LINEAR)
    
    # Perform OCR on the preprocessed image
    results = ocr.ocr(binary_image, cls=True)
    
    # Extract decimal numbers using regex and convert them to floats
    decimal_numbers = [
        float(re.sub(',', '.', match))  # Convert commas to dots and convert to float
        for line in results
        for res in line
        for match in re.findall(r'\b\d+\.\d+\b', res[1][0])  # Extract decimal numbers with regex
    ]
    
    # Calculate and return the sum of all detected decimal numbers, rounded to 2 decimal places
    total_sum = round(sum(decimal_numbers), 2)
    
    return total_sum
