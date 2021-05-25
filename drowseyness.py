import playsound
import cv2
import numpy as np
import imutils
from imutils import face_utils
from imutils.video import VideoStream
from threading import Thread
import argparse
import time
import tensorflow as tf
from keras.preprocessing import image
import os, time
# alarm function
def sound_alarm(path="alert_signal.mp3"):
	playsound.playsound(path)
		



# ap =argparse.ArgumentParser()
# ap.add_argument("-a", "--alarm", help="G:\\python course\\drowsy\\alert_signal.mp3")
# args = vars(ap.parse_args())



def detection(face_cas_path="haarcascade_frontalface_default.xml", leyepath="haarcascade_lefteye_2splits.xml", reyepath="haarcascade_righteye_2splits.xml"):
	model = tf.keras.models.load_model("drowiness_new6.h5")
	IMG_SIZE = 145
	face_cascade = cv2.CascadeClassifier(face_cas_path)
	leye = cv2.CascadeClassifier(leyepath)
	reye = cv2.CascadeClassifier(reyepath)
	# eye_cas = cv2.CascadeClassifier(eye_cas_path)
	# cap = VideoStream(src=args["webcam"]).start()
	cap = cv2.VideoCapture(0)
	start_time = time.time()
	count = 0
	EYE_FRAME = 48
	AlARM = False
	while True:
		_, frame = cap.read()
		color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		faces = face_cascade.detectMultiScale(color, 1.3, 5)
		left_eye = leye.detectMultiScale(color)
		# eye_cascade = eye_cas.detectMultiScale(color)
		
		for (x, y, w, h) in faces:
			cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
			roi_color = frame[y:y+h, x:x+w]
			roi_color = cv2.resize(roi_color, (IMG_SIZE, IMG_SIZE))
			image_pixels = image.img_to_array(roi_color)
			image_pixels = np.expand_dims(image_pixels, axis=0)
			image_pixels /= 255.0
			prediction = model.predict(image_pixels)
			max_index = np.argmax(prediction[0])
			labels = ("yawn", "no_yawn", "Closed", "Open")
			predict_emo = labels[max_index]			
			cv2.putText(frame, predict_emo, (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

			for(ex, ey, ew, eh) in left_eye:
				cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)
				roi_color2 = frame[ey:ey+eh, ex:ex+ew]
				roi_color2 = cv2.resize(roi_color2, (IMG_SIZE, IMG_SIZE))
				image_pixels2 = image.img_to_array(roi_color2)
				image_pixels2 = np.expand_dims(image_pixels2, axis=0)
				image_pixels2 /= 255.0
				prediction2 = model.predict(image_pixels2)
				max_index2 = np.argmax(prediction2[0])
				predict_emo2 = labels[max_index2]
				window_large = cv2.resize(roi_color2, (224, 224))
				cv2.putText(frame, predict_emo2, (int(x)-50, int(y)-50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
				if (max_index2 == 2):
					count+=1
					print(count)
					if count >= EYE_FRAME:
						if not AlARM:
							AlARM = True
							sound_alarm()
				else:
					count = 0
					AlARM = False


				cv2.imshow("eye", window_large)
		cv2.imshow("img", frame)		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

		

# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	detection()
	# alarm()

