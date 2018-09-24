import fire
import os
import json
import numpy as np

from svm import train
from evaluating import evaluate
from modules.utils import chunks


def remove_predicted_files(predicted_dir='../output/predicted/'):
    for i in os.listdir(predicted_dir):
        os.unlink(predicted_dir + i)


def tunning(id_chunk):
    manga_ids = [i.split('.')[0] for i in os.listdir('../output/train/')]
    manga_ids_chunks = list(chunks(manga_ids, 32))
    c_params = [10**i for i in np.arange(-6, 6.1, 0.25)]
    gamma_params = [10**i for i in np.arange(-10, 10.1, 0.25)]
    predicted_dir = '../output/predicted/%d/' % id_chunk

    total_loop = len(c_params) * len(gamma_params)
    loop_cnt = 0

    print('total loop: %d' % total_loop)

    test_ids = manga_ids_chunks[id_chunk]
    train_ids = [i for i in manga_ids if i not in test_ids]

    for c_param in c_params:
        for gamma_param in gamma_params:
            print('\n###############')
            print('tunning on [id_chunk, c, gamma]: [{}, {}, {}]'.format(id_chunk, c_param, gamma_param))
            print('###############\n')

            remove_predicted_files(predicted_dir)
            train(train_ids, test_ids, c_param, gamma_param, predicted_dir)
            result = evaluate(test_ids, predicted_dir)

            result = {
                'test_id': id_chunk,
                'c': float(c_param),
                'gamma': float(gamma_param),
                'p': result['p'],
                'r': result['r'],
                'f': result['f'],
            }

            with open('../output/result/tunning_result_chunk_%d_loop_%d.json' % (id_chunk, loop_cnt), 'w') as fp:
                json.dump(result, fp)
            fp.close()

            loop_cnt += 1
            print('finished: %d / %d' % (loop_cnt, total_loop))


if __name__ == '__main__':
    fire.Fire(tunning)
