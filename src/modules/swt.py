def swt(edges, sobel_x, sobel_y, direction, mag, height, width):
    rays = []
    swt = np.full(edges.shape, np.Infinity)

    # Ignore error from divide by zero
    np.seterr(divide='ignore', invalid='ignore')

    step_x_g, step_y_g = sobel_x, sobel_y

    gradient_x_init = np.divide(step_x_g, MAGNITUDE)
    gradient_y_init = np.divide(step_y_g, MAGNITUDE)

    for y in range(height):
        for x in range(width):
            if edges[y][x] != 0:
                cur_x, cur_y = x, y
                gradient_x, gradient_y = gradient_x_init, gradient_y_init

                ray = [{'x': x, 'y': y}]
                i = 0

                while True:
                    i += 1
                    try:
                        cur_x = math.floor(x + gradient_x * i)
                        cur_y = math.floor(y + gradient_y * i)
                    except ValueError:
                        # Catch Nan value when currenly position reached to outside a image
                        break

                    try:
                        ray.append({'x': cur_x, 'y': cur_y})
                        if edges[cur_y][cur_x] != 0:
                            gradient_value = gradient_x * -gradient_x_init[cur_y, cur_x] + gradient_y * -gradient_y_init[cur_y, cur_x]
                            
                            if abs(abs(round(np.degrees(direction[y, x])) - round(np.degrees(direction[cur_y, cur_x]))) - 180) < 90:
                                thickness = math.sqrt( (cur_x - x) * (cur_x - x) + (cur_y - y) * (cur_y - y) )
                                rays.append(ray)

                                for pos in ray:
                                    if min(thickness, swt[pos['y'], pos['x']]) > 50:
                                        continue
                                    swt[pos['y'], pos['x']] = min(thickness, swt[pos['y'], pos['x']])
                            break
                    except IndexError:
                        break

    for ray in rays:
        median = np.median([swt[pos['y'], pos['x']] for pos in ray])
        for pos in ray:
            swt[pos['y'], pos['x']] = min(median, swt[pos['y'], pos['x']])
        
    return swt