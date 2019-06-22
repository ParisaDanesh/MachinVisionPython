import cv2
import glob
import math
# 155

bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

sampleImgs =  [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in
                sorted(glob.glob('./0_simple number/*.PNG'), reverse=False)]

# im = cv2.imread('./1_Scale Number/0.PNG', cv2.IMREAD_UNCHANGED)
im = cv2.imread('./2_Rotation Number/9.PNG', cv2.IMREAD_UNCHANGED)
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
_, thresh1 = cv2.threshold(im, 160, 255, cv2.THRESH_BINARY_INV)
M = cv2.moments(thresh1)
huM = cv2.HuMoments(M)
for i in range(0, 7):
    huM[i] = -1 * math.copysign(1.0, huM[i]) * math.log10(abs(huM[i]))

tmp = []
for img in sampleImgs:
    _, thresh = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY_INV)
    sampleMoments = cv2.moments(thresh)
    sampleHuMoments = cv2.HuMoments(sampleMoments)
    for i in range(0, 7):
        sampleHuMoments[i] = -1 * math.copysign(1.0, sampleHuMoments[i]) * math.log10(abs(sampleHuMoments[i]))

    s= abs(huM)-abs(sampleHuMoments)
    s=  [x**2 for x in s]
    s= sum(s)
    s = [math.sqrt(x) for x in s]
    tmp.append(s)

dist = tmp.index(min(tmp))
print(dist)