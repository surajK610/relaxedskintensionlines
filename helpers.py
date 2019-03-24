import numpy as np
from scipy import interpolate



def average(pt1, pt2, weights=None):
    return np.average([pt1, pt2], axis=0, weights=weights)

def interpolate_pts(pts, kind='quadratic', on=True, range=1):
    x = pts[:, 0]
    y = pts[:, 0]
    if x[-1] - x[0] == 0 & y[-1] - y[0] ==0:
        return pts
    elif ((x[-1] - x[0] ) > (y[-1] - y[0])):
        return interpolate_pts_x(pts, kind, on, range)
    else:
        return interpolate_pts_y(pts, kind, on, range)




def interpolate_pts_y(pts, kind='quadratic', on=True, range=1):
    y = pts[:, 0]
    x = pts[:, 1]
    if (x[-1] < x[0]):
        if y[-1] > y[0]:
            return interpolate_pts_x(pts, kind, on, range)
        else:
            return pts
    if on:
        try:
            f = interpolate.interp1d(x, y, kind=kind)
            x_new = np.arange(x[0], x[-1], range)
            y_new = f(x_new)
            pts_new = np.zeros(shape=(x_new.shape[0], 2), dtype=np.int32)
            pts_new[:, 1] = x_new
            pts_new[:, 0] = y_new
        except:
            pts_new = pts
    else:
        return pts
    return pts_new


def interpolate_pts_x(pts, kind='quadratic', on=True, range=1):
    x = pts[:, 0]
    y = pts[:, 1]
    if (x[-1] < x[0]):
        if y[-1] > y[0]:
            return interpolate_pts_y(pts, kind, on, range)
        else:
            return pts
    if on:
        try:
            f = interpolate.interp1d(x, y, kind=kind)
            x_new = np.arange(x[0], x[-1], range)
            y_new = f(x_new)
            pts_new = np.zeros(shape=(x_new.shape[0], 2), dtype=np.int32)
            pts_new[:, 0] = x_new
            pts_new[:, 1] = y_new
        except:
            pts_new = pts
    else:
        return pts
    return pts_new

def spline_x(pts, on=True, range=1):
    x = pts[:, 0]
    y = pts[:, 1]
    if on:
        try:
            x_new = np.arange(x[0], x[-1], range)
            y_new =  interpolate.spline(x, y, x_new)
            pts_new = np.zeros(shape=(x_new.shape[0], 2), dtype=np.int32)
            pts_new[:, 0] = x_new
            pts_new[:, 1] = y_new
        except:
            pts_new = pts
    else:
        return pts
    return pts_new

def spline_y(pts, on=True, range=1):
    y = pts[:, 0]
    x = pts[:, 1]
    if on:
        try:
            x_new = np.arange(x[0], x[-1], range)
            y_new =  interpolate.spline(x, y, x_new)
            pts_new = np.zeros(shape=(x_new.shape[0], 2), dtype=np.int32)
            pts_new[:, 1] = x_new
            pts_new[:, 0] = y_new
        except:
            pts_new = pts
    else:
        return pts
    return pts_new

# def interpolate_pts_x(pts, kind='quadratic', on=True, range=1):
#     return pts
#
# def interpolate_pts_y(pts, kind='quadratic', on=True, range=1):
#     return pts

def three_pts_btw(pt1, pt2, on=True):
    midpoint = np.average([pt1, pt2], axis=0)
    quad1 = np.average([pt1, midpoint], axis=0)
    quad3 = np.average([midpoint, pt2], axis=0)
    return (quad1[0], quad1[1]), (midpoint[0], midpoint[1]), (quad3[0], quad3[1])

#determines whether right side or left side face is facing (from onlooker that is facing the person)
def determine_side(chin, right_eye, left_eye):
    chin_left = chin[0]
    chin_right = chin[16]
    eye_left = left_eye[0]
    eye_right = right_eye[3]
    left_side_distance = ((chin_left[0] - eye_left[0])**2 + (chin_left[1] - eye_left[1])**2)**.5
    right_side_distance = ((chin_right[0] - eye_right[0])**2 + (chin_right[1] - eye_right[1])**2)**.5

    if left_side_distance > right_side_distance:
        return 'left'
    elif right_side_distance > left_side_distance:
        return 'right'
    else:
        return 'front'