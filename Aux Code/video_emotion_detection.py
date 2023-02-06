# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 18:07:07 2021

@author: WILLY
"""

import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
import face_recognition


#Capture video from default camera
webcam_video_stream = cv2.VideoCapture(0) 

#load model and load the weights
face_exp_model = model_from_json(open("code/dataset/facial_expression_model_structure.json", "r").read())
face_exp_model.load_weights("code/dataset/facial_expression_model_weights.h5")

#Declare emotions label
emotions_label = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

#Initialize array var to hold all face locations in the frame
all_face_locations = []

while True:
    
    ret,current_frame = webcam_video_stream.read()
    
    current_frame_small = cv2.resize(current_frame, (0,0), fx=0.25,fy=0.25)
    
    all_face_locations = face_recognition.face_locations(current_frame_small, model='hog')
    
    cv2.imshow("Webcam Video", current_frame)
    
    for index,current_face_location in enumerate(all_face_locations):
        
        top_pos, right_pos, bottom_pos, left_pos = current_face_location
        
        top_pos = top_pos * 4
        right_pos =  right_pos * 4
        bottom_pos = bottom_pos * 4
        left_pos = left_pos * 4
        
        print('Found face {} at top:{}, right:{}, bottom:{}, left:{}'.format(index+1, top_pos,right_pos,bottom_pos,left_pos))
        
        #Extract face from frame
        current_face_image = current_frame[top_pos:bottom_pos,left_pos:right_pos]
        
        #preprocess input, convert it to an image like the data in the dataset
        #Convert to grayscale
        current_face_image = cv2.cvtColor(current_face_image, cv2.COLOR_BGR2GRAY)
        
        #Resize to 48x48px
        current_face_image = cv2.resize(current_face_image, (48, 48))
        
        #convert PIL image to 3D numpy array
        img_pixels = image.img_to_array(current_face_image)
        
        #expand the shape of array into single row multiple columns
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        
        #normalize pixels in range [0, 255] to scale of [0, 1]
        img_pixels /=255
        
        #do predictions using model , get predictions values for 7 expressions
        exp_predictions = face_exp_model.predict(img_pixels)
        #find max indexed prediction value (0-7)
        max_index = np.argmax(exp_predictions[0])
        #get label from emotions_label
        emotion_label = emotions_label[max_index]
        
        #display name as text in image
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, emotion_label, (left_pos,bottom_pos), font, 0.5, (255,255,255),1)
        #Draw rectangle on face detected
        cv2.rectangle(current_frame, (left_pos,top_pos), (right_pos,bottom_pos), (255, 255, 0), 1)
        
        
    cv2.imshow("Webcam Video", current_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
webcam_video_stream.release()
cv2.destroyAllWindows()
        