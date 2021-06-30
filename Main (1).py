from cv2 import cv2
import os
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
import time
import random


model = model_from_json(open("/home/devil/Desktop/Mood/fer.json", "r").read())

model.load_weights('/home/devil/Desktop/Mood/fer.h5')


face_haar_cascade = cv2.CascadeClassifier('/home/devil/Desktop/Mood/haarcascade_frontalface_default (1).xml')


cap=cv2.VideoCapture(0)
now = time.time()
future = now + 15
while True:
    ret,test_img=cap.read()         
    test_img=cv2.flip(test_img,1,0)
    if not ret:
        continue   
    gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)


    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
        roi_gray=gray_img[y:y+w,x:x+h]         
        roi_gray=cv2.resize(roi_gray,(48,48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad','surprise','neutral')
        predicted_emotion = emotions[max_index]

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow('Main Window ',resized_img)
        key = cv2.waitKey(30)& 0xff
     
        if time.time() > future:
            try:
                
                     cv2.destroyAllWindows()
                     cap.release()

                     if predicted_emotion == "happy":
                         mc= ['/home/devil/Music/']
                         pm=random.choice(mc)
                         os.startfile(pm)
                
                     if predicted_emotion == "sad":
                         mc= ['F:\\Final year project\\song\\sad1.mp3','F:\\Final year project\\song\\sad2.mp3','F:\\Final year project\\song\\sad3.mp3','F:\\Final year project\\song\\sad4.mp3','F:\\Final year project\\song\\sad5.mp3','F:\\Final year project\\song\\sad6.mp3','F:\\Final year project\\song\\sad7.mp3']
                         pm=random.choice(mc)
                         os.startfile(pm)

                     if predicted_emotion == "fear":
                         mc=['F:\\Final year project\\song\\fear1.mp3']
                         pm=random.choice(mc)
                         os.startfile(pm)
                         break
                    
            except:
                print('Please stay focus in Camera frame atleast 15 seconds & run again this program:)')
                break

            
        if key == 27:
            break
    


    