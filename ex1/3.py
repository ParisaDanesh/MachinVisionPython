import cv2
import numpy as np

img = cv2.imread("./q2.png")
cv2.imshow("ORIGINAL", img)

b, g, r = cv2.split(img)

colors = {'red':r, 'green':g, 'blue':b}
rgb = {'red':(0,0,255), 'green':(0,255,0), 'blue':(255,0,0)}

for key, value in colors.items():
    res = np.where(value==255)
    top_left= (np.min(res[1]),np.min(res[0]))
    bottom_right= (np.max(res[1]), np.max(res[0]))
    cv2.rectangle(img, top_left, bottom_right, (0,255,255),3)
    cv2.putText(img,key, (np.min(res[1]),np.min(res[0])-5), cv2.FONT_HERSHEY_COMPLEX, 0.75,rgb[key],1)

cv2.imshow("DETECTED COLORS", img)
cv2.waitKey(-100)
cv2.destroyAllWindows()