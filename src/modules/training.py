from sklearn import svm
from sklearn import preprocessing
from sklearn import neighbors, datasets


def train(x, y, x_test):
    scalar = preprocessing.StandardScaler().fit(x)
    x_scaled = scalar.transform(x)
    x_test_scaled = scalar.transform(x_test)

    clf = svm.SVC(C=10**1.18)
    model = clf.fit(x_scaled, y)
    return clf.predict(x_test_scaled)

    # clf = neighbors.KNeighborsClassifier()
    # model = clf.fit(x_scaled, y)
    # return clf.predict(x_test_scaled)
