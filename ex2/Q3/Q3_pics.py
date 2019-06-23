import cv2
import glob
import numpy as np

filenames= glob.glob('./hand_pic/*.jpg')
imgs= [cv2.imread(img) for img in filenames]

flag=1
for img in imgs:
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,thresh= cv2.threshold(gray_img,220,255,cv2.THRESH_BINARY_INV)
    thresh = cv2.erode(thresh, kernel=np.ones((3,3), np.uint8), iterations=3)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if cnts:
        hand = max(cnts, key=cv2.contourArea)
        x_COG = int(np.average([x[0][0] for x in hand]))
        y_COG = int(np.average([y[0][1] for y in hand]))

        # cv2.drawContours(img,hand, -1, (0,0,255), 3)
        max_dist = max((np.linalg.norm([x_COG - x[0][0], y_COG - x[0][1]]) for x in hand))

        cv2.circle(thresh, (x_COG, y_COG), int(max_dist * 0.6), (0, 255, 255), -1)

        fingers, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        numFingers = len(fingers) - 1
        cv2.putText(img, "fingers: "+str(numFingers), (10,30),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
        cv2.imwrite(str(flag)+'.jpg', img)
        cv2.imshow('hey',img)
        flag+=1
        cv2.waitKey(0)
        cv2.destroyAllWindows()







