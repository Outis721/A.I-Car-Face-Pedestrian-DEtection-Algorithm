import cv2



# LOAD PRETRAINED DATA FROM OPEN CV Algorithm)
import xml.etree.ElementTree as ET

xmlfile = "car_detect.xml"
tree = ET.parse("car_detect.xml")
root = tree.getroot()

#car Data Classifier
#car_data = "car_detect.xml"

#img
img_file  = "WhatsApp Image 2022-03-15 at 8.09.38 AM.jpeg"


#Create Open Cv image
img = cv2.imread(img_file)

#Convert to GreyScale
black_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Create Car Classifier
car_tracker = cv2.CascadeClassifier("car_detect.xml")

#Detect Cars
cars = car_tracker.detectMultiScale(black_white)

# Draw Rectangle
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255))









#Display Cars Found
cv2.imshow("OutisDetect", img)

cv2.waitKey()





print("code completed")