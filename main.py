import cv2

#LOAD PRETRAINED DATA FROM OPEN CV Algorithm
import xml.etree.ElementTree as ET

xmlfile = "face_detector.xml"
tree = ET.parse('face_detector.xml')
root = tree.getroot()

#TRAINED FACE
trained_face_data = cv2.CascadeClassifier("face_detector.xml")

#IMPORT IMAGES TO DETECT FACE
#img = cv2.imread("00001.png")

#capture Webcam
webcam = cv2.VideoCapture(0)

#ITERATE 4EVA
while True:
    # Read Current Frame
    success_frame_read, frame = webcam.read()
    # CONVERT TO GRAY SCALE
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # DETECT FACE
    face_coordinate = trained_face_data.detectMultiScale(grayscaled_img)
    # Draw Rectangle
    for (x, y, w, h) in face_coordinate:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 256, 0))
        # CODE TO SHOW IMG
        cv2.imshow("OutiFaceDetect", frame)
        # WAIT KEY
        cv2.waitKey(1)

    print("code completed")
