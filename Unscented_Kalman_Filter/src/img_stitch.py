# find longitude and latitude of each pixel in the image
# convert (lambda, fi, 1) to Cartesian (built in function)
# use estimate rotate quaternions to map position into world frame
# project into cylinder
import numpy as np
from scipy import io
import cv2
import os
import math
import pylab as pl
from matplotlib import pyplot as plt

# normalize time line
def time_norm(t_cam, t_rot):
    t_cam = (t_cam[0] - t_cam[0, 0]).tolist()
    t_rot = (t_rot[0] - t_rot[0, 0]).tolist()

    total = len(t_cam)
    length = len(t_rot)
    rot_ind = [0]
    for i in range(1, total):
        pivot = t_cam[i]
        pre = math.fabs(t_rot[-1] - pivot)
        target = t_rot[-1]

        for k in range(rot_ind[-1] + 1, length):
            if math.fabs(t_rot[k] - pivot) >= pre:
                break
            else:
                pre = math.fabs(t_rot[k] - pivot)
                target = k

        rot_ind.append(target)
    return rot_ind


# cart to sph coordinate
def cart2sph(x, y, z):
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return azimuth, elevation, r


# sph to cart coordinate
def sph2cart(azimuth, elevation, r):
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return x, y, z


# img to sph coordinate
def pos_map(row, col):
    # meshgrid map
    nx, ny = (col, row)
    x = np.linspace(0, col - 1, nx)
    y = np.linspace(0, row - 1, ny)
    xv, yv = np.meshgrid(x, y)

    xv = xv.reshape(76800, 1)
    yv = yv.reshape(76800, 1)

    longti = (159 - xv) * (np.pi / 3 / 320)
    lati = (119 - yv) * (np.pi / 4 / 240)

    x, y, z = sph2cart(longti, lati, 1)
    return x, y, z, yv, xv


# rotation coordinate
def map2world(rot, x, y, z):
    x = x.reshape(-1, 76800)
    y = y.reshape(-1, 76800)
    z = z.reshape(-1, 76800)
    pos = np.zeros((3, 76800))

    pos[0, :] = x
    pos[1, :] = y
    pos[2, :] = z

    world = np.dot(rot, pos)
    return world[0, :], world[1, :], world[2, :]


# stitch image
def project(img, rot, rot_index):
    x, y, z, yv, xv = pos_map(240, 320)

    time = len(rot_index)
    im_sti = np.zeros((960, 1920, 3)) # new panorama

    for i in range(time):
        im_i = img[:, :, :, i]
        ind = rot_index[i]
        rot_i = rot[ind]

        X, Y, Z = map2world(rot_i, x, y, z)
        az, ele, r = cart2sph(X, Y, Z)

        theta = (az + np.pi) / (np.pi / 3 / 320)
        height = 960 - (ele + (np.pi / 2)) / (np.pi / 4 / 240)

        theta = np.floor(theta)
        height = np.floor(height)

        for k in range(76800):
            new_x = theta[k]
            new_y = height[k]
            if new_x < 0 or new_y < 0:
                continue
            ori_x = xv[k, 0]
            ori_y = yv[k, 0]

            im_sti[new_y, new_x, :] = im_i[ori_y, ori_x, :]

    return im_sti

# main code
def main():
    folder = "cam1"  # the folder stores all cam datas
    for mat in os.listdir(folder):
        cams = io.loadmat(os.path.join(folder, mat))
        camm = cams["cam"]
        ts_cam = cams["ts"]

        ts_imu = np.load("imu_ts.npy")[8].transpose()  # the file stores time information
        est = np.load('vicon_rots.npy')[8]  # the file stores estimated rotations

        rot_index = time_norm(ts_cam, ts_imu)  # normalize time line
        im_res = project(camm, est, rot_index) # stitch images
        pl.imshow(im_res)
        plt.show()
main()







