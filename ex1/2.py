import cv2


def callback(x):
    pass

cv2.namedWindow("cap")

cap = cv2.VideoCapture(0)
while(True):
    ret,frame=cap.read()
    if ret==False:
        break
    cv2.createTrackbar("test", "cap", 0, 30, callback)
    value= cv2.getTrackbarPos('test','cap')
    if value==10:
        cv2.imwrite("laugh.jpg",frame)
    elif value==20:
        cv2.imwrite("sad.jpg", frame)
    elif value==30:
        cv2.imwrite("anger.jpg", frame)
    cv2.imshow('cap', frame)
    cv2.waitKey(5)
    if cv2.waitKey(30)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()