
import cv2
from random import randrange

# loads some pre-trained data on face frontal from opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# loads webcam video
webcam = cv2.VideoCapture(0)
while True:

    # read returns frame
    successful_frame_read, frame = webcam.read()

    # grey scale the frame
    grey_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detects coordinates
    face_coordinates = trained_face_data.detectMultiScale(grey_scale)

    # de-structure coordinates and print rectangle
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

    # show image
    cv2.imshow('Face Detector', frame)
    key = cv2.waitKey(1)

    # to break loop
    if key == 81 or key == 113:
        break

webcam.release()
