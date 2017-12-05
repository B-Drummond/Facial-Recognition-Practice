# -*- coding: utf-8 -*-
"""
Name: Brianna Drummond

Internet Sources: 
    http://docs.opencv.org/trunk/dd/d43/tutorial_py_video_display.html
    http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html
    https://realpython.com/blog/python/face-detection-in-python-using-a-webcam/
"""

import cv2

videoCapt = cv2.VideoCapture(0) #Set video source to webcam

#Using haar cascade classifier for facial recognition
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret, frame = videoCapt.read() #reads frame from video source
    
    #Convert frame from BGR color space to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect the faces
    faces = faceCascade.detectMultiScale(
        gray, #Image (frame)
        scaleFactor=1.3, #Reduction of image size at each scale
        minNeighbors=5, #Minimum number of neighbors for each rectangle
        minSize=(30, 30) #Minimum object size for detection    
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    #Show the frame after altering with the facial recognition rectangle code
    cv2.imshow('frame', frame) 
    
    #Exit script when 'q' is entered
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture when finished
videoCapt.release()
cv2.destroyAllWindows()
