import cv2

import numpy as np


webcam = cv2.VideoCapture(0)

#Pedestrian and car Detect
face_data = cv2.CascadeClassifier("face_detector.xml")
smile_data = cv2.CascadeClassifier("smile.xml")

while True:
    # Frame read
    successful_frame_read, frame = webcam.read()
    #Abort if Error
    if not successful_frame_read:
        break

    #Convert to Grayscale
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # DETECT FACE
    faces = face_data.detectMultiScale(frame_grayscale)

    # Draw on Face frame
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 200, 50))

    #Get Face
    the_face = frame[y:y+h, x:x+w]

    face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

    smile = smile_data.detectMultiScale(face_grayscale, scaleFactor=1.7, minNeighbors=20)
    # label smile in Face
    if len(smile) > 0:
        cv2.putText(frame, "Smile_Joy", (x, y+h+40), fontScale=2, fontFace=cv2.FONT_HERSHEY_TRIPLEX, color=(255, 255, 255)  )

    # SHow Frame
    cv2.imshow("Outis_DEtect", frame)

    # Wait key
    cv2.waitKey(1)

print("code complete")
