import cv2
import numpy as np
from matplotlib import pyplot as plt

from module.getSwt import get_swt
from module.connectComponents import connect_components
from module.findLetters import find_letters
from module.findWords import find_words

OUTPUT_FOLDER = 'output/'

class MangaCleaner:
    def __init__(self, path):
        src = cv2.imread(path)
        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

        height, width = src.shape[:2]

        # Get edgeMa
        edges = cv2.Canny(src_gray, 150, 240)

        # Get Magnitude and gradient angel
        invert_gray_image = cv2.bitwise_not(src_gray)
        sobel_x = cv2.Sobel(invert_gray_image, cv2.CV_64F, 1, 0)
        sobel_y = cv2.Sobel(invert_gray_image, cv2.CV_64F, 0, 1)

        mag, angle = cv2.cartToPolar(sobel_x, sobel_y)

        swt = get_swt(edges, sobel_x, sobel_y, angle, mag, height, width)
        print('# Getting SWT is done.')
        shapes = connect_components(swt)
        print('# Connect component is done.')
        swts, heights, widths, topleft_pts, images = find_letters(swt, shapes)
        print('# Finding letters is done.')
        seperate_word_images = find_words(swts, heights, widths, topleft_pts, images)
        print('# Finding words is done.')

        masks = []
        for word in seperate_word_images:
            mask = np.zeros(swt.shape, dtype=np.uint8)
            for char in word:
                mask += char
            masks.append(mask)

        bounding_points = []

        for mask in masks:
            # self.showImg('Mask image', mask, 2)

            x_min_pos, x_max_pos = None, None
            y_min_pos, y_max_pos = None, None

            im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                min_xs = tuple(contour[contour[:,:,0].argmin()][0])[0]
                max_xs = tuple(contour[contour[:,:,0].argmax()][0])[0]
                min_ys = tuple(contour[contour[:,:,1].argmin()][0])[1]
                max_ys = tuple(contour[contour[:,:,1].argmax()][0])[1]

                if x_min_pos == None or min_xs < x_min_pos:
                    x_min_pos = min_xs
                if x_max_pos == None or max_xs > x_max_pos:
                    x_max_pos = max_xs

                if y_min_pos == None or min_xs < y_min_pos:
                    y_min_pos = min_ys
                if y_max_pos == None or max_ys > y_max_pos:
                    y_max_pos = max_ys

            bounding_points.append(((x_min_pos, y_min_pos), (x_max_pos, y_max_pos)))

        bounding_img = src.copy()
        bounding_img1 = src.copy()

        for point in bounding_points:
            cv2.rectangle(bounding_img, point[0], point[1], (0, 0, 255), 1)

        for i in range(len(topleft_pts)):
            cv2.rectangle(bounding_img1, (topleft_pts[i][1], topleft_pts[i][0]), (topleft_pts[i][1] + widths[i], topleft_pts[i][0] + heights[i]), (0, 0, 255), 1)

        # Display
        # self.showImg('Source image', src)
        # self.showImg('Edges image', swt, 2)
        self.showImg('Bounding image', bounding_img, 2)
        # self.showImg('Result image', result)

        plt.show()

    def showImg(self, title, img, flag=0):
        f, ax = plt.subplots()
        f.canvas.set_window_title(title)

        if flag == 0:
            ax.imshow(img, interpolation = 'none')
        elif flag == 1:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), interpolation = 'none')
        elif flag == 2:
            ax.imshow(img, cmap='gray', interpolation = 'none')
