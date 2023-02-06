# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 20:25:47 2021

@author: WILLY
"""

#importing rquired libraries

import cv2
import dlib
import face_recognition

print(cv2.__version__)
print(dlib.__version__)
print(face_recognition.__version__)

#Loading image to detect
#detect faces in the image
path = r'D:/Code-py/images/testing/trump-modi-unknown.jpg'

original_image = cv2.imread(path)
name_of_person = "Unknown face"

#load the sample images and get the 128 embeddings from them
modi_image = face_recognition.load_image_file('images/samples/modi.jpg')
modi_face_encodings = face_recognition.face_encodings(modi_image)[0]

trump_image = face_recognition.load_image_file('images/samples/trump.jpg')
trump_face_encodings = face_recognition.face_encodings(trump_image)[0]


#save encodings and corresponding labels in separeate arrays in same order
known_face_encodings = [modi_face_encodings, trump_face_encodings]
known_face_names = ["Donald Trump", "Naredra Modi", "Unknown face", "Unknown face", "Unknown face"]

#Load unknown image to recognize face in it
image_to_recognize = face_recognition.load_image_file('images/testing/trump-modi-unknown.jpg')

#Detect all face s in the image
all_face_locations = face_recognition.face_locations(image_to_recognize, model='hog')
#Detect face encodings for faces detected
all_face_encodings = face_recognition.face_encodings(image_to_recognize, all_face_locations) 


#print no of faces detected
print('There are {} no of faces in this image'.format(len(all_face_locations)))

#looping through face locations
for current_face_location, current_face_encoding in zip(all_face_locations, all_face_encodings):
    
    #splitting the tuple to get the four position values of current face
    top_pos,right_pos,bottom_pos,left_pos = current_face_location
    
    #printing location of current face
   # print('Found face {} at top:{},right:{},bottom:{},left:{}'.format(index+1, top_pos,right_pos,bottom_pos,left_pos))
    
    #Find matches and get list of all matches
    all_matches = face_recognition.compare_faces(all_face_encodings, current_face_encoding)
    #String to hold the label
    name_of_person = "Unknown face"
    
    
    #Check if all matches have atleast one item
    #If yes, get index of face located in the first index of all_matches
    #Get name corresponding to index number and save it in name_of_person
    if True in all_matches:
        first_match_index = all_matches.index(True)
        name_of_person = known_face_names[first_match_index]
        
            
   
    
    
    #Draw rectangle around the face
    cv2.rectangle(original_image, (left_pos,top_pos),(right_pos,bottom_pos), (255,0,0),1)
    
    #Display corresponding text in the image
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(original_image, name_of_person, (left_pos,bottom_pos+20), font, 0.5,(255,255,255),1)
    
    cv2.imshow("Faces identified", original_image)
    
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)