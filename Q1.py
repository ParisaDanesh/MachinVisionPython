import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

def unpickle(file):
    # unpickles cifar10 dataset
    with open(file, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic


def getCIFAR10(direc, filename, batches):
    # Converts the data in batches to a full training set
    for j in range(1, batches+1):
        file = direc + filename + str(j)
        dic = unpickle(file)
        if j == 1:
            X_train = dic[b'data']
            y_train = dic[b'labels']
        else:
            temp_X = dic[b'data']
            temp_y = dic[b'labels']
            X_train = np.concatenate((X_train, temp_X))
            y_train = np.concatenate((y_train, temp_y))
    return X_train, y_train

direc = './cifar-10-batches-py/'
test_file = 'test_batch'
filename = 'data_batch_'
X_train, y_train = getCIFAR10(direc, filename, 5)
data_test = unpickle(direc + test_file)
X_test = data_test[b'data']
y_test = data_test[b'labels']


clf = LinearSVC()
clf.fit(X_train,y_train)


# clf = KNeighborsClassifier(n_neighbors=5)
# clf.fit(X_train, y_train)
print('accuracy: {}'.format(clf.score(X_test, y_test)))
