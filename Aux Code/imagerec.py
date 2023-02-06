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
    print('Found face {} at top:{},right:{},bottom:{},left:{}'.format(index+1, top_pos,right_pos,bottom_pos,left_pos))
    
    #slicing current face from image
    current_face_image = image_to_detect[top_pos:bottom_pos,left_pos:right_pos]
    
    #showing current face with dynamic title
    cv2.imshow("Face no"+str(index+1), current_face_image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)