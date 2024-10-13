import numpy as np
from ultralytics import YOLO
import os
import cv2

def apartment_type(result):
    indxs = list()
    if len(result.boxes.cls) != 0:
        for cls in result.boxes.cls:
            print(cls)
            indxs.append(int(cls.item()))

    room_types = ['Balcony', 'BalconyDoor', 'BalconyRailing', 'Bathroom', 'Bedroom', 'DiningRoom', 'Garage', 'Kitchen', 'KitchenCountertop', 'Living Room', 'Master Bedroom', 'Office', 'PrayerRoom', 'Staircase', 'StoreRoom', 'StudyRoom', 'Utility', 'Wardrobe', 'balcony-door', 'balcony-railing', 'garage', 'kitchen', 'kitechenCountertop', 'office', 'staircase', 'wardrobe']
    live_rooms = ['Bedroom', 'DiningRoom', 'Living Room', 'Master Bedroom', 'Office', 'PrayerRoom', 'StudyRoom']

    apartment_type = ''
    c=0

    for i in indxs:
        if room_types[i] in live_rooms:
            c += 1

    if c == 0:
        print('error, c = 0')
    
    elif c == 1:
        apartment_type = 'однушка'
    
    elif c == 2:
        apartment_type = 'двушка'

    elif c == 3:
        apartment_type = 'трёшка'

    elif c == 4:
        apartment_type = '4-х комнатная'

    elif c == 5:
        apartment_type = '5-и комнатная'
    
    else: 
        apartment_type = 'дворец'

    return apartment_type, c
