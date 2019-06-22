import cv2
import glob
import imutils
import numpy as np

imgs = [cv2.imread(img) for img in
        sorted(glob.glob('./images/*.jpg'), reverse=False)]

for img in imgs:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sobelX = cv2.Sobel(gray,  cv2.CV_64F, dx=1, dy=0, ksize=-1)
    sobelY = cv2.Sobel(gray,  cv2.CV_64F, dx=0, dy=1, ksize=-1)

    gradient = cv2.subtract(sobelX, sobelY)
    gradient = cv2.convertScaleAbs(gradient)

    blur_img = cv2.blur(gradient, (9,9))
    _, thresh = cv2.threshold(blur_img, 230,255, cv2.THRESH_BINARY)

    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((7, 21)))
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)

    cnts =cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    max_cnts = max(cnts, key= cv2.contourArea)

    maxRect = cv2.minAreaRect(max_cnts)
    box = cv2.boxPoints(maxRect).astype(int)
    cv2.drawContours(img, [box], -1, (0, 255, 0), 3)


    cv2.imshow('barcode',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
