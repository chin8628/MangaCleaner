from sklearn import svm
from sklearn import preprocessing


def train(x, y, x_test):
    clf = svm.SVC()

    scalar = preprocessing.StandardScaler().fit(x)
    x_scaled = scalar.transform(x)
    model = clf.fit(x_scaled, y)

    x_test_scaled = scalar.transform(x_test)
    return clf.predict(x_test_scaled)
