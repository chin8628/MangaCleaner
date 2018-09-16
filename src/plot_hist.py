import matplotlib.pyplot as plt
import cv2
from modules.file_manager import load_dataset
from tqdm import tqdm

title = 'DollGun'
page = 31
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
