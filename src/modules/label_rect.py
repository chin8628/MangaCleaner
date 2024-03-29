import cv2
import matplotlib.pyplot as plt


def label(img, topleft_pts, heights, widths, color):
    for index in range(0, len(topleft_pts)):
        cv2.rectangle(
            img,
            (topleft_pts[index][1], topleft_pts[index][0]),
            (topleft_pts[index][1] + widths[index], topleft_pts[index][0] + heights[index]),
            color,
            2
        )

    plt.imshow(img)
    plt.show()
