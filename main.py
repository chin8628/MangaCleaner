import cv2
import numpy as np
from matplotlib import pyplot as plt

class MangaCleaner:

    src, src_gray = [], []
    mag, angle = [], []
    edges = []
    swt = []
    height, width = (0, 0)

    def __init__(self, path):
        self.src = cv2.imread(path)
        self.src_gray = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.src.shape[:2]
        self.swt = np.zeros((self.height, self.width), np.int16)

        # Find Magnitude and gradient angel
        invert_gray_image = cv2.bitwise_not(self.src_gray)
        sobel_x = cv2.Sobel(invert_gray_image, cv2.CV_64F, 1, 0)
        sobel_y = cv2.Sobel(invert_gray_image, cv2.CV_64F, 0, 1)
        self.mag, self.angle = cv2.cartToPolar(sobel_x, sobel_y, angleInDegrees=1)

        # Find edge
        self.edges = cv2.Canny(self.src_gray, 150, 240)

        # print (self.angle[24])
        # print (self.edges[29])

        another_edge = self.stepToEdge(27, 24, self.getDirection(self.angle[24][27]))
        print ("Another edge {x,y,step_cnt}:", another_edge)
        print ("My edge angle:", self.angle[24][27])
        print ("Another edge angle:", self.angle[another_edge['y']][another_edge['x']])

        self.getSwt()

        # Display
        # self.showImg('Contour image', contour_img)
        self.showImg('SWT image', self.swt)
        self.showImg('Edge image', self.edges)
        self.showImg('Source image', self.src)

    def connect_component(self):
        # GO SLEEP!

    def getSwt(self):
        rays = []
        for curr_y in range(self.height):
            for curr_x in range(self.width):
                another_edge = self.stepToEdge(curr_x, curr_y, self.getDirection(self.angle[curr_y][curr_x]))

                if another_edge != -1 and self.angle[curr_y][curr_x] == (self.angle[another_edge['y']][another_edge['x']] + 180 ) % 360 and self.angle[curr_y][curr_x] < 180:
                    self.swt[curr_y][curr_x] = another_edge['step_cnt']
                    rays.append((curr_x, curr_y))

        median = np.median([self.swt[y][x] for (x, y) in rays])
        for (x, y) in rays:
            self.swt[y][x] = min(median, self.swt[y][x])

    def showImg(self, title, img):
        f, ax = plt.subplots()
        f.canvas.set_window_title(title)

        if (len(str(img[0][0])) != 1 and len(img[0][0]) == 3):
            ax.imshow(img, interpolation = 'none')
        else:
            ax.imshow(img, cmap='gray', interpolation = 'none')

    def stepToEdge(self, x, y, direction, step_cnt=0):
        """ input:
                direction: [0-7]
                    0 1 2
                    7 - 3
                    6 5 4
                x: x axis's position
                y: y axis's position
            return
                (x,y) of another edge that function found
        """

        # Found boundary of image
        if x == self.width - 1 or y == self.height or x < 0 or y < 0:
            return -1

        add_step_direction = {
            0 : {'add_x': -1, 'add_y': -1},
            1 : {'add_x': 0, 'add_y': -1},
            2 : {'add_x': 1, 'add_y': -1},
            3 : {'add_x': 1, 'add_y': 0},
            4 : {'add_x': 1, 'add_y': 1},
            5 : {'add_x': 0, 'add_y': 1},
            6 : {'add_x': -1, 'add_y': 1},
            7 : {'add_x': -1, 'add_y': 0},
        }

        new_x, new_y = x + add_step_direction[direction]['add_x'], y + add_step_direction[direction]['add_y']

        if (x < self.width - 1 and y < self.height - 1 and self.edges[new_y][new_x] == 0):
            return self.stepToEdge(new_x, new_y, direction, step_cnt + 1)

        return {'x': x, 'y': y, 'step_cnt': step_cnt}

    def getDirection(self, angle):
        if (angle > 337.5 or angle <= 22.5):
            return 3
        elif (angle > 22.5 and angle <= 67.5):
            return 4
        elif (angle > 67.5 and angle <= 112.5):
            return 5
        elif (angle > 112.5 and angle <= 157.5):
            return 6
        elif (angle > 157.5 and angle <= 202.5):
            return 7
        elif (angle > 202.5 and angle <= 247.5):
            return 0
        elif (angle > 247.5 and angle <= 292.5):
            return 1
        elif (angle > 292.5 and angle <= 337.5):
            return 2

mangaCleaner = MangaCleaner('1.jpg')
plt.show()