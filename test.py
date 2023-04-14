import cv2
import numpy as np

img = cv2.imread("./digit1.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

erosion = cv2.dilate(thresh, kernel= np.ones((3,3), np.uint8), iterations=2)

cnts, _= cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in cnts:
    M = cv2.moments(c)
    perimeter = cv2.arcLength(c, True)

    if M['m00'] > 200:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(img, '{}'.format(round(perimeter, 2)), (x - 10, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (0, 255, 0), 1)

cv2.imshow("org", img)
cv2.waitKey(0)