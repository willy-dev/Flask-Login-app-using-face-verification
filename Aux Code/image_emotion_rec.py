# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 20:25:47 2021

@author: WILLY
"""

#importing rquired libraries


import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
import face_recognition

print(cv2.__version__)
#print(dlib.__version__)
print(face_recognition.__version__)

#Loading image to detect
#detect faces in the image
path = r'D:/Code-py/images/testing/trump-modi.jpg'

image_to_detect = cv2.imread(path)

#load model and load the weights
face_exp_model = model_from_json(open("code/dataset/facial_expression_model_structure.json", "r").read())
face_exp_model.load_weights("code/dataset/facial_expression_model_weights.h5")

#Declare emotions label
emotions_label = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')


all_face_locations = face_recognition.face_locations(image_to_detect, model='hog')

#print no of faces detected
print('There are {} no of faces in this image'.format(len(all_face_locations)))



    #looping through face locations
for index, current_face_location in enumerate(all_face_locations):
    
    #splitting the tuple to get the four position values of current face
    top_pos,right_pos,bottom_pos,left_pos = current_face_location
    
    #printing location of current face
    print('Found face {} at top:{},right:{},bottom:{},left:{}'.format(index+2, top_pos,right_pos,bottom_pos,left_pos))
    
    
    
    #slicing current face from image
    current_face_image = image_to_detect[top_pos:bottom_pos,left_pos:right_pos]
    #Draw rectangle on face detected
    cv2.rectangle(image_to_detect, (left_pos,top_pos), (right_pos,bottom_pos), (255, 255, 0), 2) 
    
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
    cv2.putText(image_to_detect, emotion_label, (left_pos,bottom_pos), font, 0.5, (255,255,255),1)
    
        
        
cv2.imshow("Image emotions", image_to_detect)
    
    
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)