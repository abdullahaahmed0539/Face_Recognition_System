import cv2

# Loads some pre-trained data on face frontal from opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# choose an image to detect faces in
img = cv2.imread('assets/RDJ.jpg')

# shows the img
cv2.imshow('Face Detector', img)
cv2.waitKey()

