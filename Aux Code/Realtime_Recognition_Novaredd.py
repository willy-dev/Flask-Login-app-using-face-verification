# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 16:43:25 2021

@author: WILLY
"""

import face_recognition
import os
import cv2

def known():
    known_faces_dir = r'D:/Code-py/images/samples'
    known_faces = []
    known_names = []
    
    for name in os.listdir(known_faces_dir):
        for filename in os.listdir(f'{known_faces_dir}/{name}'):
            image = face_recognition.load_image_file(f'{known_faces_dir}/ {name}/ {filename}')
            encoding = face_recognition.face_encodings(image)[0]
            known_faces.append(encoding)
            known_names.append(name)
    return known_faces, known_names

def get_match():
    known_faces, known_names = known()
    video = cv2.VideoCapture(0)
    TOLERANCE = 0.5
    #frame_thickness = 3
    #font_thickness = 2
    model = 'hog'
    img_counter = 0
    while True:
        ret,image = video.read()
        if not ret:
            print("Failed to grab frame")
            break
        locations = face_recognition.face_locations(image, model = model)
        encodings = face_recognition.face_encodings(image, locations)
        
        for face_encoding, face_location in zip(encodings, locations):
            results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
            match = None
            
            try:
                match = known_names[results.index(True)]
            except ValueError:
                print("Not Recognized")
            else:
                print(f"match found:{match}")
                img_counter += 1
        if img_counter==1:
            break
    video.release()
    return match

    

