# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 18:07:07 2021

@author: WILLY
"""

import cv2
import face_recognition
import numpy as np

webcam_video_stream = cv2.VideoCapture(1)

#load the sample images and get the 128 embeddings from them
modi_image = face_recognition.load_image_file('images/samples/modi.jpg')
modi_face_encodings = face_recognition.face_encodings(modi_image)[0]

trump_image = face_recognition.load_image_file('images/samples/trump.jpg')
trump_face_encodings = face_recognition.face_encodings(trump_image)[0]

mareh_image = face_recognition.load_image_file('images/samples/marete.jpg')
mareh_face_encodings = face_recognition.face_encodings(mareh_image)[0]

coco_image2 = face_recognition.load_image_file('images/samples/coco.jpg')
coco_face_encodings2 = face_recognition.face_encodings(coco_image2)[0]

isika_image = face_recognition.load_image_file('images/samples/isika.jpg')
isika_face_encodings = face_recognition.face_encodings(isika_image)[0]

kelvin_image = face_recognition.load_image_file('images/samples/kevo.jpg')
kelvin_face_encodings = face_recognition.face_encodings(kelvin_image)[0]


#save encodings and corresponding labels in separeate arrays in same order
known_face_encodings = [modi_face_encodings, trump_face_encodings, mareh_face_encodings, coco_face_encodings2, isika_face_encodings, kelvin_face_encodings]
known_face_names = ["Naredra Modi", "Donald Trump","Wilfred Marete", "Coco", "Isika Joe", "Kelvin Mwirigi"]

#Initiliaze array variable to hold all face locations, encodings, names
all_face_locations = []
all_face_encodings = []
all_face_names = []
process_this_frame = True


while True:
    
    ret,current_frame = webcam_video_stream.read()
    
    current_frame_small = cv2.resize(current_frame, (0,0), fx=0.25,fy=0.25)
    
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = current_frame_small[:, :, ::-1]
    
    ## Only process every other frame of video to save time
    if process_this_frame:
            
        #Perform image detection
        all_face_locations = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample=1, model='hog')
        
        #copy image from face recognition
        all_face_encodings = face_recognition.face_encodings(rgb_small_frame, all_face_locations)
       
        all_face_names = []
        for current_face_encoding in all_face_encodings:
        
           
            #Find matches and get list of all matches
            all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding)
            #String to hold the label
            name_of_person = "Unknown face"
            
            
            #Check if all matches have atleast one item
            #If yes, get index of face located in the first index of all_matches
            #Get name corresponding to index number and save it in name_of_person
            #if True in all_matches:
                #first_match_index = all_matches.index(True)
               ##name_of_person = known_face_names[first_match_index]
                #else:
            # print('There are {} no of faces in this image', [first_match_index])
            
            #Or instead, use the known face with smallest distance to known face
            face_distances = face_recognition.face_distance(known_face_encodings, current_face_encoding)
            best_match_index = np.argmin(face_distances)
            if all_matches[best_match_index]:
                name_of_person = known_face_names[best_match_index]
            
            all_face_names.append(name_of_person)
            
    process_this_frame = not process_this_frame
    
    #Dispaly results
    
    for current_face_location, name in zip(all_face_locations, all_face_names):
        
        #splitting the tuple to get the four position values of current face
        top_pos,right_pos,bottom_pos,left_pos = current_face_location
        #Change position of magnitude to fit actual size video frame
        top_pos = top_pos * 4
        right_pos =  right_pos * 4
        bottom_pos = bottom_pos * 4
        left_pos = left_pos * 4
        
        
        #Draw rectangle around the face
        cv2.rectangle(current_frame, (left_pos,top_pos),(right_pos,bottom_pos), (255,0,0),1)
        
        #Display corresponding text in the image
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, name_of_person, (left_pos,bottom_pos+20), font, 0.5,(255,255,255),1)
        
    
    cv2.imshow("Faces identified", current_frame)
    
            
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
            
    
webcam_video_stream.release()
cv2.destroyAllWindows()
        