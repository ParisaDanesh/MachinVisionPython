import cv2
import os.path

def cvtColor(img):
    return 0.299*img[:,:,2]+0.587*img[:,:,1]+0.114*img[:,:,0]

def plotImg(img):
    for item in img:
        indexNum = str(img.index(item))
        cv2.namedWindow(indexNum)
        cv2.imshow(indexNum, item)
    cv2.waitKey(-100)
    cv2.destroyAllWindows()

while True:
    imgPath = input("enter your image path(q to exit): ")

    if imgPath == 'q':
        exit(0)
    elif not os.path.exists(imgPath):
        print("incorrect path, Try again\n")
        continue
    else:
        img = cv2.imread(imgPath)
        tmp = cvtColor(img)
        dispImg = [img, tmp]
        plotImg(dispImg)
