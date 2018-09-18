import os
from modules.file_manager import load_dataset, save_by_dict

import matplotlib.pyplot as plt
import cv2
from modules.file_manager import load_dataset
from tqdm import tqdm

title = 'AosugiruHaru'
page = 12
img = cv2.imread('../../Dataset_Manga/Manga109/images/%s/%03d.jpg' % (title, page))
data = load_dataset('../output/%s-%03d.json' % (title, page))

for datum in tqdm(data):
    # plt.bar(range(0, 256), datum['hist'])
    # plt.savefig('../output_hist/{}/{}-hist.png'.format(datum['is_text'], datum['id']))
    # plt.close()

    cv2.imwrite('../output_hist/{}/{}.jpg'.format(datum['is_text'], datum['id']), img[
        datum['topleft_pt']['y']:datum['topleft_pt']['y'] + datum['height'],
        datum['topleft_pt']['x']:datum['topleft_pt']['x'] + datum['width']
    ])

key = input('continue? ...')
while key != 'y':
    key = input('continue? ...')

fileName = '%s-%03d.json' % (title, page)
data = load_dataset('../output/' + fileName)

img_text = list(map(
    lambda x: int(x.split('.')[0]),
    filter(
        lambda x: x.split('.')[1] == 'jpg',
        os.listdir('../output_hist/1/')
    )
))
img_not_text = list(map(
    lambda x: int(x.split('.')[0]),
    filter(
        lambda x: x.split('.')[1] == 'jpg',
        os.listdir('../output_hist/0/')
    )
))

for datum in data:
    if datum['id'] in img_text:
        datum['is_text'] = 1
    elif datum['id'] in img_not_text:
        datum['is_text'] = 0

save_by_dict('../output/verified/' + fileName, data)


def clear_img(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


clear_img('../output_hist/0/')
clear_img('../output_hist/1/')
