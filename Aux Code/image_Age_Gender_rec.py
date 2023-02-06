# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 20:25:47 2021

@author: WILLY
"""

#importing rquired libraries


import cv2
import face_recognition

print(cv2.__version__)
#print(dlib.__version__)
print(face_recognition.__version__)

#Loading image to detect
#detect faces in the image
path = r'D:/Code-py/images/testing/trump-modi.jpg'

image_to_detect = cv2.imread(path)



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
    cv2.rectangle(image_to_detect, (left_pos,top_pos), (right_pos,bottom_pos), (255, 255, 0), 1) 
    
    #Age and gender mean values calculated by numpy.mean()
    AGE_GENDER_MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    #Create blob of current face slice
    current_face_image_blob = cv2.dnn.blobFromImage(current_face_image, 1, (227, 227), AGE_GENDER_MODEL_MEAN_VALUES, swapRB=False)
    
    #Declare gender labels, protext and caffemodel file paths
    gender_label_list = ['Male', 'Female']
    gender_protext = "code/dataset/gender_deploy.prototxt"
    gender_caffemodel = "code/dataset/gender_net.caffemodel"
    
    #Create model from files and provide blob as input
    gender_cov_net = cv2.dnn.readNet(gender_caffemodel, gender_protext)
    gender_cov_net.setInput(current_face_image_blob)
    
    #Get gender predictions 
    gender_predictions = gender_cov_net.forward()
    #Pass index to label array to get label text
    gender = gender_label_list[gender_predictions[0].argmax()]
    
    #predict age
    #Declare the labels
    age_label_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    age_protext = "code/dataset/age_deploy.prototxt"
    age_caffemodel = "code/dataset/age_net.caffemodel"
    
    #create the model
    age_cov_net = cv2.dnn.readNet(age_protext, age_caffemodel)
    age_cov_net.setInput(current_face_image_blob)
    
    #Get age predictions
    age_predictions = age_cov_net.forward()
    #Pass index to label array to get label text
    age = age_label_list[age_predictions[0].argmax()]
        
    #display name as text in image
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image_to_detect, gender+" "+age+"yrs", (left_pos,bottom_pos+20), font, 0.5, (255,255,255),1)
    
        
cv2.imshow("Image Age and Gender", image_to_detect)
    
    
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)