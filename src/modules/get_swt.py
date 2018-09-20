import math
import numpy as np
import logging


def get_swt(edges, sobel_x, sobel_y, direction, magnitude, height, width) -> np.ndarray:
    rays = []
    swt = np.full(edges.shape, np.Infinity)

    # Ignore error from divide by zero
    np.seterr(divide='ignore', invalid='ignore')

    step_x_g, step_y_g = sobel_x, sobel_y

    gradient_x_init = np.divide(step_x_g, magnitude)
    gradient_y_init = np.divide(step_y_g, magnitude)

    for y in range(height):
        for x in range(width):
            if edges[y][x] == 0:
                continue

            gradient_x = gradient_x_init[y, x]
            gradient_y = gradient_y_init[y, x]

            ray = [{'x': x, 'y': y}]
            i = 0

            while True:
                i += 1

                try:
                    cur_x = math.floor(x + gradient_x * i)
                    cur_y = math.floor(y + gradient_y * i)

                except ValueError:
                    # Catch Nan value when currently position reached to outside a image
                    break

                try:
                    ray.append({'x': cur_x, 'y': cur_y})

                    # IF still not found the another edge THEN go to next step
                    if edges[cur_y][cur_x] == 0:
                        continue

                    # Difference between the direction of start edge and another edge
                    # IF difference is exceed 90degree THEN ignore this ray
                    if abs(abs(round(np.degrees(direction[y, x])) - round(np.degrees(direction[cur_y, cur_x]))) - 180) > 60:
                        break

                    thickness = math.sqrt((cur_x - x) * (cur_x - x) + (cur_y - y) * (cur_y - y))
                    rays.append(ray)

                    for pos in ray:
                        swt[pos['y'], pos['x']] = min(thickness, swt[pos['y'], pos['x']])

                    break

                except IndexError:
                    break

    for ray in rays:
        median = np.median([swt[pos['y'], pos['x']] for pos in ray])

        for pos in ray:
            swt[pos['y'], pos['x']] = min(median, swt[pos['y'], pos['x']])

    logging.getLogger(__name__).info('Finished.')

    return swt
