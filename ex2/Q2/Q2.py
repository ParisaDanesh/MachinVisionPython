import cv2

cv2.namedWindow('Video')
cap = cv2.VideoCapture('eye.mp4')
# cap = cv2.VideoCapture(0)

while True:
    ret, frame= cap.read()
    if ret is False:
        break

    rows, cols, _ = frame.shape
    frame = frame[rows//5:rows*2//3 , cols//3:cols*3//4]
    nrows, ncols, _ = frame.shape

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _,thresh_frame = cv2.threshold(gray_frame, 4, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(thresh_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if cnts :
        max_cnt = max(cnts, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(max_cnt)

        if x+w//2 > 290:
            txt = "dir: Right"
        elif x+w//2 < 250:
            txt = "dir: Left"
        else:
            txt = "dir: Center"

        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.putText(frame, txt, (15,30), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255,255,255))
        cv2.line(frame, (x+w//2,0), (x+w//2,nrows), (0,255,0), 2)
        cv2.line(frame, (0,y+h//2), (ncols,y+h//2), (0,255,0), 2)

    else:
        cv2.putText(frame, 'not detected', (15,30), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0,0,255))

    cv2.imshow('Video',frame)
    cv2.waitKey(30)
    if cv2.waitKey(30) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
