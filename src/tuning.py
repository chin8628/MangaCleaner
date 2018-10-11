from svm_train import train
from svm_test import test
from evaluating import evaluate

import json
import multiprocessing
import os


def main(c_param, gamma_param):
    model_name = train(c_param, gamma_param)
    test(model_name)
    fmeasure = evaluate('../output/predicted/%s/' % model_name)

    print(model_name)

    data = {
        'c': c_param,
        'gamma': gamma_param,
        'fmeasure': fmeasure,
    }

    with open('../output/result/%s.json' % model_name, 'w') as fp:
        json.dump(data, fp)
    fp.close()

    os.remove('./model/%s.pkl', model_name)


if __name__ == '__main__':
    c_params = [0, 1, 2, 3, 4, 5]
    gamma_params = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]

    for c_idx in range(len(c_params)):
        jobs = []
        for gamma_idx in range(len(gamma_params)):
            p = multiprocessing.Process(target=main, args=(c_params[c_idx], gamma_params[gamma_idx]))
            p.start()
            jobs.append(p)

        for i in jobs:
            i.join()
