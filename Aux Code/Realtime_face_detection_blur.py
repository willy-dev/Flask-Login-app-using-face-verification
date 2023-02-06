# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 18:07:07 2021

@author: WILLY
"""

import cv2
import face_recognition

webcam_video_stream = cv2.VideoCapture(0)

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
        
        #Slicing current image from current frame
        current_face_image = current_frame[top_pos:bottom_pos,left_pos:right_pos]
        
        #Age and gender mean values calculated by numpy.mean()
        AGE_GENDER_MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        #Create blob of current face slice
        current_face_image_blob = cv2.dnn.blobFromImage(current_face_image, 1, (227, 227), AGE_GENDER_MODEL_MEAN_VALUES, swapRB=False)
        
        #Declare gender labels, protext and caffemodel file paths
        gender_label_list = ['Male', 'Female']
        gender_protext = "code/dataste/gender_deploy.prototxt"
        gender_caffemodel = "code/dataset/gender_net.caffemodel"
        
        #Create model from files and provide blob as input
        gender_cov_net = cv2.dnn.readNet(gender_caffemodel, gender_protext)
        gender_cov_net.setInput(current_face_image_blob)
        
        
        
        #Draw rectangle on face detected
        cv2.rectangle(current_frame, (left_pos,top_pos), (right_pos,bottom_pos), (255, 255, 0), 1)
        
        
    cv2.imshow("Webcam Video", current_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
webcam_video_stream.release()
cv2.destroyAllWindows()
        