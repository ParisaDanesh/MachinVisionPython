import cv2
import numpy as np

# cv2.namedWindow('Video')
# cv2.namedWindow('First_frame')
# cv2.namedWindow('difference')
cv2.namedWindow('finaly')

cap= cv2.VideoCapture('back.mp4')

_,first_frame= cap.read()
resized_first_frame= cv2.resize(first_frame,(700,500))
first_gray= cv2.cvtColor(resized_first_frame, cv2.COLOR_BGR2GRAY)


while True:
    ret, frame = cap.read()
    if ret is False:
        break

    resized_frame1= cv2.resize(frame,(700,500))
    resized_frame= cv2.resize(frame,(700,500))
    gray_frame= cv2.cvtColor(resized_frame,cv2.COLOR_BGR2GRAY)

    difference = cv2.absdiff(gray_frame, first_gray)
    _,thresh_difference = cv2.threshold(difference,50,255, cv2.THRESH_BINARY)

    erosion = cv2.erode(thresh_difference, kernel=np.ones((3,3), np.uint8), iterations=5)
    dilation = cv2.dilate(erosion, kernel=np.ones((3,3), np.uint8), iterations=7)

    tmp = np.nonzero(dilation)
    if len(tmp[0]) > 0 and len(tmp[1]) > 0:
        cv2.rectangle(resized_frame, (min(tmp[1]), min(tmp[0])), (max(tmp[1]), max(tmp[0])), (0, 255, 0), 2)

    # cv2.imshow('First_frame',resized_first_frame)
    # cv2.imshow('Video',resized_frame1)
    # cv2.imshow('difference',difference)
    # cv2.imshow('dilation',dilation)
    # cv2.imshow('erosion',erosion)
    cv2.imshow('finaly',resized_frame)



    cv2.waitKey(30)
    if cv2.waitKey(5)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
