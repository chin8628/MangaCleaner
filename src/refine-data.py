import cv2

from modules.file_manager import load_dataset, save_by_dict

img = cv2.imread('../../Dataset_Manga/Manga109/images/AppareKappore/005.jpg')

filename = 'Appare-005.json'
data = load_dataset('../output/raw/' + filename)

verified_no_text_data = []
cnt = 0
cnt_data_0 = len(list(filter(lambda x: x['is_text'] == 0, data)))
for datum in filter(lambda x: x['is_text'] == 0, data):
    cv2.namedWindow("Image")
    cv2.imshow('Image',
               img[
                   datum['topleft_pt']['y']:datum['topleft_pt']['y'] +
                   datum['height'],
                   datum['topleft_pt']['x']:datum['topleft_pt']['x'] +
                   datum['width']
               ]
               )

    print('Is this a text? [y/n]')
    ans = cv2.waitKey()
    cv2.destroyAllWindows()
    print(chr(ans), '{} / {}'.format(cnt, cnt_data_0))
    cnt += 1

    if chr(ans) == 'n':
        verified_no_text_data.append(datum)
    elif chr(ans) == 'q':
        quit()
    else:
        pass

data = list(filter(lambda x: x['is_text'] == 1, data)) + verified_no_text_data
save_by_dict('../output/verified/{}'.format(filename), data)
