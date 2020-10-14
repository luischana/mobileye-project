from operator import itemgetter
import numpy as np


def calc_TFL_dist(prev_container, curr_container, focal, pp):
    norm_prev_pts, norm_curr_pts, R, foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp)

    if abs(tZ) < 10e-6:
        print('tz = ', tZ)

    elif norm_prev_pts.size == 0:
        print('no prev points')

    elif norm_prev_pts.size == 0:
        print('no curr points')

    else:
        curr_container.corresponding_ind, curr_container.traffic_lights_3d_location, curr_container.valid = calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ)

    return curr_container


def prepare_3D_data(prev_container, curr_container, focal, pp):
    norm_prev_pts = normalize(prev_container.traffic_light, focal, pp)
    norm_curr_pts = normalize(curr_container.traffic_light, focal, pp)
    R, foe, tZ = decompose(np.array(curr_container.EM))

    return norm_prev_pts, norm_curr_pts, R, foe, tZ


def calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ):
    norm_rot_pts = rotate(norm_prev_pts, R)

    pts_3D = []
    corresponding_ind = []
    validVec = []

    for p_curr in norm_curr_pts:
        corresponding_p_ind, corresponding_p_rot = find_corresponding_points(p_curr, norm_rot_pts, foe)
        Z = calc_dist(p_curr, corresponding_p_rot, foe, tZ)
        valid = (Z > 0)

        if not valid:
            Z = 0

        validVec.append(valid)
        P = Z * np.array([p_curr[0], p_curr[1], 1])
        pts_3D.append((P[0], P[1], P[2]))
        corresponding_ind.append(corresponding_p_ind)

    return corresponding_ind, np.array(pts_3D), validVec


# transform pixels into normalized pixels using the focal length and principle point
def normalize(pts, focal, pp):
    return (pts - pp) / focal


# transform normalized pixels into pixels using the focal length and principle point
def unnormalize(pts, focal, pp):
    return (pts * focal) + pp


# extract R, foe and tZ from the Ego Motion
def decompose(EM):
    R = EM[:3, :3]
    t = EM[:3, 3]
    tx, ty, tz = t[0], t[1], t[2]
    foe = np.array([tx/tz, ty/tz])

    return R, foe, tz


# rotate the points - pts using R
def rotate(pts, R):
    ones = np.ones((3, 1), int)
    rotate_mat = np.dot(R, (np.hstack([pts, ones])).T)

    return (rotate_mat[:2]/rotate_mat[2]).T


def find_corresponding_points(p, norm_pts_rot, foe):
    # compute the epipolar line between p and foe
    m = (foe[1] - p[1]) / (foe[0] - p[0])
    n = (p[1] * foe[0] - foe[1] * p[0]) / (foe[0] - p[0])

    # run over all norm_pts_rot and find the one closest to the epipolar line
    distance = [[abs((m * p_rot[0] + n - p_rot[1]) / np.sqrt(pow(m, 2) + 1)), i] for i, p_rot in enumerate(norm_pts_rot)]

    # return the closest point and its index
    index, val = min(distance, key=itemgetter(0))

    return index, norm_pts_rot[index]


def calc_dist(p_curr, p_rot, foe, tZ):
    # calculate the distance of p_curr using x_curr, x_rot, foe_x and tZ
    Zx = (tZ * (foe[0] - p_rot[0])) / (p_curr[0] - p_rot[0])

    # calculate the distance of p_curr using y_curr, y_rot, foe_y and tZ
    Zy = (tZ * (foe[1] - p_rot[1])) / (p_curr[1] - p_rot[1])

    # combine the two estimations and return estimated Z
    dx = abs(foe[0] - p_curr[0])
    dy = abs(foe[1] - p_curr[1])

    ratio = dx / (dx + dy)

    return Zx * ratio + Zy * (1 - ratio)
