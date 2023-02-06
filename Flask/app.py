from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np
app=Flask(__name__)


webcam_video_stream = cv2.VideoCapture(0)

#load the sample images and get the 128 embeddings from them
""""
modi_image = face_recognition.load_image_file('images/samples/modi.jpg')
modi_face_encodings = face_recognition.face_encodings(modi_image)[0]

trump_image = face_recognition.load_image_file('images/samples/trump.jpg')
trump_face_encodings = face_recognition.face_encodings(trump_image)[0]

coco_image2 = face_recognition.load_image_file('images/samples/coco.jpg')
coco_face_encodings2 = face_recognition.face_encodings(coco_image2)[0]
"""""
mareh_image = face_recognition.load_image_file('Marete/marete.jpg')
mareh_face_encodings = face_recognition.face_encodings(mareh_image)[0]

isika_image = face_recognition.load_image_file('Joe/isika.jpg')
isika_face_encodings = face_recognition.face_encodings(isika_image)[0]

kelvin_image = face_recognition.load_image_file('Kelvin/kevo.jpg')
kelvin_face_encodings = face_recognition.face_encodings(kelvin_image)[0]


#save encodings and corresponding labels in separeate arrays in same order
known_face_encodings = [mareh_face_encodings, isika_face_encodings, kelvin_face_encodings]
known_face_names = ["Wilfred Marete", "Isika Joe", "Kelvin Mwirigi"]

#Initiliaze array variable to hold all face locations, encodings, names
all_face_locations = []
all_face_encodings = []
all_face_names = []
process_this_frame = True

def gen_frames():  
    while True:
        
        ret,frame = webcam_video_stream.read()

        if not ret:
            break
        else:
            current_frame_small = cv2.resize(frame, (0,0), fx=0.25,fy=0.25)
            
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = current_frame_small[:, :, ::-1]
            
            ## Only process every other frame of video to save time
            #if process_this_frame:
                    
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
                
            #process_this_frame = not process_this_frame
            
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
                cv2.rectangle(frame, (left_pos,top_pos),(right_pos,bottom_pos), (255,0,0),1)
                
                #Display corresponding text in the image
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name_of_person, (left_pos,bottom_pos+20), font, 0.5,(255,255,255),1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True)