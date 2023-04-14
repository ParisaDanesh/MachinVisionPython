# How to run
# python final.py -i input(digit1.jpg)

import argparse
import time
import cv2
from pathlib import Path
from imutils.object_detection import non_max_suppression
from pyimagesearch.helpers import pyramid
from pyimagesearch.helpers import sliding_window
from sklearn import datasets
from sklearn.svm import SVC
import numpy as np
from skimage.feature import hog
from sklearn.externals import joblib


def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(6, image.shape[0]-6, stepSize):
		for x in range(6, image.shape[1]-6, stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])




if not Path('saved_SVC.pkl').exists():
    mnist_dataset = datasets.fetch_openml('mnist_784')
    features = np.array(mnist_dataset.data, 'uint8')
    labels = np.array(mnist_dataset.target, 'int')
    hog_list = []

    for img in features:
        hog_list.append(hog(img.reshape(28,28), visualize=False))


    clf = SVC(probability=True, kernel='rbf')
    clf.fit(hog_list, labels)
    print('training Done!')

    joblib.dump(clf,'saved_SVC.pkl')

else:

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    args = vars(ap.parse_args())

    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # load the image and define the window width and height
    image = cv2.imread(args["image"])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    (winW, winH) = (128, 128)

    clf = joblib.load('saved_SVC.pkl')
    rects = []

    for (x, y, window) in sliding_window(gray, stepSize=10, windowSize=(winW, winH)):  # step_size=10

        boundings = []

        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        resize_window = cv2.resize(window, (28, 28), interpolation=cv2.INTER_AREA)

        # preprocess
        _, thresh = cv2.threshold(resize_window, 180, 255, cv2.THRESH_BINARY_INV)

        # erosion = cv2.dilate(thresh, kernel=np.ones((3, 3), np.uint8), iterations=2)

        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = 0
        if cnts:
            mc = max(cnts, key = cv2.contourArea)
            M = cv2.moments(mc)
            if M['m00'] > 20:
                perimeter = cv2.arcLength(mc, True)

        hog_window = hog(resize_window)

        pred_label = clf.predict(np.array([hog_window]))

        prob = clf.predict_proba(np.array([hog_window]))

        if max(prob[0]) > 0.985:
            print(prob)
            max_prob = round(max(prob[0]), 2)
            # print(max_prob)
            print(pred_label)
            rects.append([x, y, x + winW, y + winH, pred_label, perimeter, max_prob*100])

    rects = np.array(rects)

    pick = non_max_suppression(rects, probs= None, overlapThresh=0.39)
    for (xA, yA, xB, yB, number, per, prob) in pick:
        if per > 44 and per != 49:
            print(prob)
            cv2.rectangle(image, (xA, yA), (xB, yB), (0,255,0), 2)
            cv2.putText(image, '{}'.format(int(number)), (xA, yA),
                        cv2.FONT_HERSHEY_COMPLEX, 1.5, (0,0,255),3)
            cv2.putText(image, '{}'.format(int(prob)), (xA, yA+133),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)

    cv2.imwrite("./result.jpg", image)
    cv2.imshow("image", image)
    cv2.waitKey(0)
