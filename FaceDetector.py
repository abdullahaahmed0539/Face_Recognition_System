import cv2
from random import randrange

# loads some pre-trained data on face frontal from opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# choose an image to detect faces in
img = cv2.imread('assets/cp.jpg')


# grey scale the img
grey_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect faces
face_coordinates = trained_face_data.detectMultiScale(grey_scale)


for (x, y, w, h) in face_coordinates:
    # draw rectangle on the img
    cv2.rectangle(img, (x, y), (x + w, y + h), (randrange(128,256), randrange(128,256), randrange(128,256)), 4)

print(face_coordinates)

# shows the img
cv2.imshow('Face Detector', img)
cv2.waitKey()

