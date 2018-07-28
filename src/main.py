import logging
from pathlib import Path

import cv2
import fire
import matplotlib.pyplot as plt

# Modules
from text_detection import text_detection


class Main:
    def text_detection(self, path):
        expected_height = 1200

        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        image_file = Path(str(path))
        acceptable_types = ['.jpg', '.JPG', '.jpeg', '.JPEG']

        logger.info('Input path: %s', path)
        logger.info('Absolute path: %s', image_file.resolve())

        if not image_file.is_file() or image_file.suffix not in acceptable_types:
            logger.error('File is not in %s types.', acceptable_types)
            quit()

        src = cv2.imread(path)
        if src.shape[0] > int(expected_height):
            src = cv2.resize(
                cv2.imread(path),
                None,
                fx=expected_height / src.shape[0],
                fy=expected_height / src.shape[0]
            )

        swt_values, heights, widths, topleft_pts, letter_images = text_detection(src, src.shape[0])

        # letter_values -> {
        #     'swt_values': swts,
        #     'heights': heights,
        #     'widths': widths,
        #     'topleft_pts': topleft_pts,
        #     'letter_images': images
        # }

        output = src.copy()

        for index in range(0, len(topleft_pts)):
            topleft_pt = topleft_pts[index]

            cv2.rectangle(
                output,
                (topleft_pt[1], topleft_pt[0]),
                (topleft_pt[1] + widths[index], topleft_pt[0] + heights[index]),
                (255, 0, 0),
                1
            )

        plt.imshow(output)
        plt.show()
        cv2.waitKey(0)


if __name__ == '__main__':
    fire.Fire(Main)
