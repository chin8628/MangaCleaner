import matplotlib.pyplot as plt
import cv2
from modules.file_manager import load_dataset
from tqdm import tqdm

img = cv2.imread(
    '../../Dataset_Manga/Manga109/images/AisazuNihaIrarenai/035.jpg')
data = load_dataset('../output/verified/Aisazu-035.json')

for datum in tqdm(filter(lambda x: x['is_text'] == 0, data)):
    plt.hist(datum['hist'], bins=list(
        range(0, 267)), range=(0, 257), density=True)
    plt.savefig(
        '../output_hist/{}/{}.png'.format(datum['is_text'], datum['id']))
    plt.close()

    cv2.imwrite('../output_hist/{}/{}_img.jpg'.format(datum['is_text'], datum['id']), img[
        datum['topleft_pt']['y']:datum['topleft_pt']['y'] + datum['height'],
        datum['topleft_pt']['x']:datum['topleft_pt']['x'] + datum['width']
    ])
