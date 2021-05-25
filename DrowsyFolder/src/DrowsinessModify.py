# important packages
import cv2
import pygame
import numpy as np
import time
import tensorflow as tf
from keras.preprocessing import image
import os, time
from time import strftime
from threading import Thread

'''
Task 1: Camera Implementation
    Using cv2

Task 2: Detection
    * Face Extract
    * Take only Face
    * Resize Extracted Face
    * Create Image to floatting array
    * Face array noralization
    * Load it in Trained Model
    * Prediction of Yawnning
    * Eye Extract
    * Take only Eye
    * Resize Extracted Eye
    * Create Image to floatting array
    * Eye array noralization
    * Load it in Trained Model
    * Prediction of Eye open open of close
Task 3: Alarm
    # Final : after counting of 15 
    # Trigger Alarm Function
        # Open eye and alarm trigger off
    # Drowsyness 
'''


class DrowsiDriver:
    def __init__(self):
        super(DrowsiDriver, self).__init__()
        self.faceCascade = "../input/haarcascade_frontalface_default.xml"
        self.eyeCascade = "../input/haarcascade_lefteye_2splits.xml"
        self.loadModel = "../input/drowsyfile1.h5" 
        self.alarmSound = "../input/alert_signal.mp3"
        self.IMG_SIZE = 145
        self.counting = 0
        self.alarm_on = False
    
    def sound_alarm(self):
    	pygame.mixer.init()
    	sound = pygame.mixer.Sound(self.alarmSound)
    	sound.play()
    	time.sleep(2)

    def camera(self):
        # load face_cascade
        face_cascade = cv2.CascadeClassifier(self.faceCascade)
        # left eye cascade
        left_eye_cascade = cv2.CascadeClassifier(self.eyeCascade)
        # load from cam
        cap = cv2.VideoCapture(0)
        return cap, face_cascade, left_eye_cascade
        

    def detection(self):
        # load model
        model = tf.keras.models.load_model(self.loadModel)
        cap, face_cascade, left_eye_cascade = self.camera()
        while True:
            _, frame = cap.read()
            frame = cv2.flip(frame, 360)
            color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_cascade.detectMultiScale(color, 1.3, 5)
            leftEye = left_eye_cascade.detectMultiScale(color, 1.3, 3)
            # detectMultiScale function is used to detect the faces. This function will return a rectangle with coordinates(x,y,w,h) around the detected face.
            # scaleFactor=1.3 specifies how much the image size is reduced with each scale. In a group photo, there may be some faces which are near the camera than others. Naturally, such faces would appear more prominent than the ones behind. This factor compensates for that
            # minNeighbours=5 specifies how many neighbours each candidate rectangle should have to retain it. You can read about it in detail here. You may have to tweak these values to get the best results. This parameter specifies the number of neighbours a rectangle should have to be called a face.
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # create rectangle
                roi_color = frame[y:y+h, x:x+w] 
                roi_color = cv2.resize(roi_color, (self.IMG_SIZE, self.IMG_SIZE)) # reduce image size
                image_pixels = image.img_to_array(roi_color) # convert to array
                image_pixels = np.expand_dims(image_pixels, axis=0) # Expand the shape of an array. Insert a new axis that will appear 
                image_pixels /= 255.0 # convert in range(0 to 1)
                prediction = model.predict(image_pixels) # predict
                maxIndex = np.argmax(prediction[0])
                # label of this project. 4 category
                labels = ["yawn", "no_yawn", "Closed", "Open"]
                time.sleep(0.02)	
                for(ex, ey, ew, eh) in leftEye: 
                    cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)
                    roi_color2 = frame[ey:ey+eh, ex:ex+ew]
                    roi_color2 = cv2.resize(roi_color2, (self.IMG_SIZE, self.IMG_SIZE))
                    image_pixels2 = image.img_to_array(roi_color2)
                    image_pixels2 = np.expand_dims(image_pixels2, axis=0)
                    image_pixels2 /= 255.0
                    prediction2 = model.predict(image_pixels2)
                    maxIndex2 = np.argmax(prediction2[0])
                    window_large = cv2.resize(roi_color2, (224, 224))
                    time.sleep(0.02)
                    if maxIndex2 == 2:
                        self.counting += 1
                        if self.counting > 15:
                            predict_emo2 = "drowsy"
                            if not self.alarm_on:
                                self.alarm_on = True # trigger alarm
                                t = Thread(target=self.sound_alarm()) # Thread for multi function run
                                t.daemon = True
                                t.start()
                            cv2.putText(frame, predict_emo2, (450, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                        else:
                            predict_emo2 = "Closed Eyes"
                            cv2.putText(frame, predict_emo2, (450, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                    else:
                        self.alarm_on = False
                        self.counting = 0
                        predict_emo = labels[maxIndex]
                        predict_emo2 = labels[maxIndex2]
                        cv2.putText(frame, predict_emo, (460, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                        cv2.putText(frame, predict_emo2, (460, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                    cv2.imshow("eye", window_large) # imsow cam for eye 
            cv2.imshow("img", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): # press 'q' to break
                break
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows() 
        # destroy all windows

if __name__ == '__main__':
    a = DrowsiDriver()
    a.detection()