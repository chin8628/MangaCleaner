import numpy as np
import math

def get_swt(edges, sobel_x, sobel_y, angle, mag, height, width):
    swt = np.empty(edges.shape)
    swt[:] = np.Infinity

    step_x_g = sobel_x
    step_y_g = sobel_y

    # Ignore error from divide by zero
    np.seterr(divide='ignore', invalid='ignore')
    
    grad_x_g = np.divide(step_x_g, mag)
    grad_y_g = np.divide(step_y_g, mag)

    rays = []

    for y in range(height):
        for x in range(width):
            if edges[y][x] != 0:
                cur_x, cur_y = x, y
                grad_x = grad_x_g[y, x]
                grad_y = grad_y_g[y, x]

                ray = [{'x': x, 'y': y}]
                i = 0

                while True:
                    i += 1
                    try:
                        cur_x = math.floor(x + grad_x * i)
                        cur_y = math.floor(y + grad_y * i)
                    except ValueError:
                        # Catch Nan value
                        break

                    try:
                        ray.append({'x': cur_x, 'y': cur_y})
                        if edges[cur_y][cur_x] != 0:
                            # Filter value which is out of domain
                            if (grad_x * -grad_x_g[cur_y, cur_x] + grad_y * -grad_y_g[cur_y, cur_x] >= -1 and
                                grad_x * -grad_x_g[cur_y, cur_x] + grad_y * -grad_y_g[cur_y, cur_x] <= 1):
                                if math.acos(grad_x * -grad_x_g[cur_y, cur_x] + grad_y * -grad_y_g[cur_y, cur_x]) < np.pi/2.0:
                                    thickness = math.sqrt( (cur_x - x) * (cur_x - x) + (cur_y - y) * (cur_y - y) )
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
    return swt