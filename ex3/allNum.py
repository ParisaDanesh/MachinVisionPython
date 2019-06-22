import cv2
import glob
from imutils import paths

sampleImgs =  [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in
                sorted(glob.glob('./0_simple number/*.jpg'), reverse=False)]

photos = []



for address in paths.list_images('./images'):
    img = cv2.imread(address)
    photos.append(img)

# print(len(photos))


for imgs in photos:
    gray = cv2.cvtColor(imgs, cv2.COLOR_BGR2GRAY)
    _, thresh1 = cv2.threshold(gray, 195, 255, cv2.THRESH_BINARY_INV)
    cnts1,_= cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt1 = min(cnts1,key= cv2.contourArea)

    d=[]

    for img in sampleImgs:
        _, thresh = cv2.threshold(img, 195, 255, cv2.THRESH_BINARY_INV)
        cnts,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt2 = min(cnts, key=cv2.contourArea)
        d.append(cv2.matchShapes(cnt1,cnt2, cv2.CONTOURS_MATCH_I2, 0))

    d_min = d.index(min(d))

    # print(d_min)

    cv2.putText(imgs, str(d_min), (7,25), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0,255,0), 1)
    cv2.imshow('predicted', imgs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()