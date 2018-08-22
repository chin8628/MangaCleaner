from sklearn import svm
from sklearn import preprocessing


def train(x, y, x_test, y_test):
    clf = svm.SVC()

    scalar = preprocessing.StandardScaler().fit(x)
    x_scaled = scalar.transform(x)
    model = clf.fit(x_scaled, y)

    x_test_scaled = scalar.transform(x_test)
    result = clf.predict(x_test_scaled)

    tp, fp, tn, fn = 0, 0, 0, 0

    for index in range(0, len(result)):
        if y_test[index] == 1 and result[index] == 1:
            tp += 1
        elif y_test[index] == 0 and result[index] == 0:
            tn += 1
        elif y_test[index] == 0 and result[index] == 1:
            fp += 1
        elif y_test[index] == 1 and result[index] == 0:
            fn += 1

    print('TP: {} FP: {} TN: {} FN: {}'.format(tp, fp, tn, fn))

    try:
        print('P: {} R: {}'.format(
            round(tp / (tp + fp), 4), round(tp / (tp + fn), 4)))
    except ZeroDivisionError:
        print('Divided by zero')

    return result
