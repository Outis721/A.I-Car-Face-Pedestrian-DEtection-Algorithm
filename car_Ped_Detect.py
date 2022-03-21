import cv2


# LOAD PRETRAINED DATA FROM Car_detect Algorithm)
import xml.etree.ElementTree as ET

xmlfile = "car_detect.xml"
tree = ET.parse("car_detect.xml")
root = tree.getroot()

# LOAD PRETRAINED DATA FROM ped_detect Algorithm)
import xml.etree.ElementTree as ET

xmlfile = "ped.xml"
tree = ET.parse("ped.xml")
root = tree.getroot()


#Pedestrian and car Detect
ped_data = cv2.CascadeClassifier("ped.xml")
car_data = cv2.CascadeClassifier("car_detect.xml")

#Get Video
video = cv2.VideoCapture("detect.mp4")



# Iterate forever over frames
while True:

    # Read the current frames
    read_successful, frame = video.read()

    if read_successful:
        # Must convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # Detect cars
    cars = car_data.detectMultiScale(grayscaled_frame)

    # Detect Pedestrians
    Pedestrians = ped_data.detectMultiScale(grayscaled_frame)

    # Draw rectangles around cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x + 2, y + 2), (x + w, y + h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Draw rectangles around pedestrians
    for (x, y, w, h) in Pedestrians:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # Display the footage with the cars & pedestrians spotted
    cv2.imshow('Self Driving Cars', frame)

    # Listen for a key press for 1 millisecond, then move on
    key = cv2.waitKey(1)

    # Stop if Q key is pressed
    if key == 81 or key == 133:
        break

# Release the VideoCapture object
video.release()



print("code completed")