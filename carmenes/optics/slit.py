import numpy as np


def slit_params_init(H, DC, dec_x, dec_y, defocus):  
    H[:, 0] += dec_x
    H[:, 1] += dec_y
    H[:, 2] += defocus
    DC[:, 0] = 0.
    DC[:, 1] = 0.
    DC[:, 2] = 1.
  
    return H, DC


def interp(x0, y0, x1, y1, n):
    if x0 == x1 and y0 < y1:
        s = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
        s_prime = s / n
        theta = 90 * np.pi / 180
        dx = s_prime * np.cos(theta)
        dy = s_prime * np.sin(theta)
    elif x0 < x1:
        m = (y0 - y1) / (x0 - x1)
        theta = np.arctan(m)
        s = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
        s_prime = s / n
        dx = s_prime * np.cos(theta)
        dy = s_prime * np.sin(theta)
    elif x0 > x1:
        m = (y0 - y1) / (x0 - x1)
        theta = np.arctan(m)
        s = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
        s_prime = s / n
        dx = -1 * s_prime * np.cos(theta)
        dy = -1 * s_prime * np.sin(theta)
    elif x0 == x1 and y0 > y1:
        s = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
        s_prime = s / n
        theta = -90 * np.pi / 180
        dx = s_prime * np.cos(theta)
        dy = s_prime * np.sin(theta)

    xaux = x0
    yaux = y0
    x_out = []
    y_out = []
    for i in range(int(n)):
        x_out.append(xaux)
        y_out.append(yaux)
        xaux += dx
        yaux += dy

    return x_out, y_out


def polygon(diameter, offset_x, offset_y, npoints):
    dtheta = 360.0 / npoints
    radius = diameter / 2.0
    theta = 0.0
    points = []
    oct_x = []
    oct_y = []
    for i in range(npoints + 1):
        oct_x.append(float(radius * np.cos(theta * np.pi / 180) + offset_x))
        oct_y.append(float(radius * np.sin(theta * np.pi / 180) + offset_y))
        theta = theta + dtheta

    x_side = []
    y_side = []
    n = 10
    for i in range(len(oct_x) - 1):
        x, y = interp(oct_x[i], oct_y[i], oct_x[i + 1], oct_y[i + 1], n)
        x_side.append(x)
        y_side.append(y)

    # print len(x_side) #number of sides
    # print len(x_side[0]) #number of points on each side
    x_final = []
    y_final = []
    coors_out = []
    for i in range(len(x_side)):
        for k in range(len(x_side[0])):
            x, y = interp(x_side[i][k], y_side[i][k], offset_x, offset_y, 20)
            x_final.append(x)
            y_final.append(y)
            coors_out.append(np.array([x, y]))

    x_final = np.array(x_final)
    y_final = np.array(y_final)
    coors_out = np.array(coors_out)

    return coors_out


def slicer(coors, d):

    vec_out = []
    for i in range(len(coors)):
        for k in range(len(coors[i][0])):

            if coors[i][1][k] > 0:
                vec_out.append(np.array([coors[i][0][k] - d / 2, coors[i][1][k] - d / 4]))
            elif coors[i][1][k] < 0:
                vec_out.append(np.array([coors[i][0][k] + d / 2, coors[i][1][k] + d / 4]))
    vec_out = np.array(vec_out)/1000

    return vec_out

