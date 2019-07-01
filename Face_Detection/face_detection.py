
#A simple face detection algorithm in python using openCV

import cv2


faces_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


def detect(gray, img):
    faces = faces_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 4)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eyes_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 255, 0), 2)
    return img


get_video = cv2.VideoCapture(0)
while True:
    _, img = get_video.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    final_op = detect(gray, img)
    cv2.imshow('Video', final_op)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
get_video.release()
cv2.destroyAllWindows()