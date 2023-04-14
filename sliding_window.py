import argparse
import time
import cv2
from pathlib import Path
from sklearn import datasets
from sklearn.svm import SVC
import numpy as np
from skimage.feature import hog
from sklearn.externals import joblib


def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
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

    # load the image and define the window width and height
    image = cv2.imread(args["image"])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (winW, winH) = (90, 128)

    clf = joblib.load('saved_SVC.pkl')

    for (x, y, window) in sliding_window(gray, stepSize=25, windowSize=(winW, winH)):

        # print(window.shape)
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        # _, thresh = cv2.threshold(window, 150, 255, cv2.THRESH_BINARY_INV)
        # cv2.imshow('h',thresh)
        # cv2.waitKey(0)
        resize_window = cv2.resize(window, (28,28), interpolation = cv2.INTER_AREA)
        hog_window = hog(resize_window)
        pred_label = clf.predict(np.array([hog_window]))
        prob = clf.predict_proba(np.array([hog_window]))

        if max(prob[0])>0.8:
            print(pred_label)
            print(max(prob[0]))
            cv2.putText(image, '{}'.format(int(pred_label)), (x-10, y-10),
                        cv2.FONT_HERSHEY_COMPLEX, 1.5, (0,255,0),3)


        clone = image.copy()
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        cv2.imshow("Window", clone)
        cv2.waitKey(1)
        time.sleep(0.025)