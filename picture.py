import cv2

# LOAD PRETRAINED DATA FROM OPEN CV Algorithm)
import xml.etree.ElementTree as ET

xmlfile = "face_detector.xml"
tree = ET.parse('face_detector.xml')
root = tree.getroot()

# TRAINED FACE
trained_face_data = cv2.CascadeClassifier("face_detector.xml")

# IMPORT IMAGES TO DETECT FACE
img = cv2.imread("00001.png")

# DETECT FACE
face_cordinates = trained_face_data.detectMultiScale(img)

# Draw Rectangle
(x, y, w, h) = face_cordinates[1]
cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255))

# Show img
cv2.imshow("OutiFaceDetect", img)
# WAIT KEY
cv2.waitKey()

print("code completed")
