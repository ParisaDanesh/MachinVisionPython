import cv2
import glob
import numpy as np

cap= cv2.VideoCapture('./hand_pic/hand_vid.mp4')
# cap= cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    if ret is False:
        break

    rows, cols, _ = frame.shape
    frame= frame[0:rows//2 , 0:cols*2//3]

    b, g, r = cv2.split(frame)
    mask= cv2.inRange(r/g, 1.55, 4)
    hand = cv2.dilate(mask, kernel=np.ones((3,3)), iterations=4)

    cnts, _ = cv2.findContours(hand, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if cnts:
        max_hand = max(cnts, key=cv2.contourArea)
        M= cv2.moments(max_hand)
        cX = int(M['m10']/M['m00'])
        cY = int(M['m01']/M['m00'])


        max_dist = max((np.linalg.norm([cX - x[0][0], cY - x[0][1]]) for x in max_hand))

        cv2.circle(hand, (cX, cY), int(max_dist*0.7), (0,255,255), -1)

        fingers, _ = cv2.findContours(hand, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        numFingers = len(fingers) - 1

        cv2.putText(frame, "fingers: "+str(numFingers), (10,30),cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)


    cv2.imshow('finger count', frame)


    if cv2.waitKey(30) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()