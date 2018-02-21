import load_data as ld
import numpy as np
import matplotlib.pyplot as plt
import MapUtils as MU
import cv2
import random

from utils import *


'''
  motion model: compute control signal based on the lidar pose
'''
def control_sig(N, pose_pre, pose_cur, yaw_pre, yaw_cur, yaw_est):
  delta_x_gb = pose_cur[0][0] - pose_pre[0][0]
  delta_y_gb = pose_cur[0][1] - pose_pre[0][1]
  delta_theta_gb = yaw_cur - yaw_pre

  delta_x_lc = np.einsum('..., ...', np.cos(yaw_pre), delta_x_gb) + np.einsum('..., ...', np.sin(yaw_pre), delta_y_gb)
  delta_y_lc = np.einsum('..., ...', -np.sin(yaw_pre), delta_x_gb) + np.einsum('..., ...', np.cos(yaw_pre), delta_y_gb)
  delta_theta_lc = delta_theta_gb

  delta_x_gb_new = (np.einsum('..., ...', np.cos(yaw_est), delta_x_lc) - np.einsum('..., ...', np.sin(yaw_est), delta_y_lc)).reshape(-1, N)
  delta_y_gb_new = (np.einsum('..., ...', np.sin(yaw_est), delta_x_lc) + np.einsum('..., ...', np.cos(yaw_est), delta_y_lc)).reshape(-1, N)
  delta_theta_gb_new = np.tile(delta_theta_lc, (1, N))

  # control signal with meters and radians unit
  ut = np.concatenate([np.concatenate([delta_x_gb_new, delta_y_gb_new], axis=0), delta_theta_gb_new], axis=0)
  return np.einsum('ji', ut)


'''
  observation model: recompute particle weights
'''
def obser_model(N, pos_phy, particles, xim, yim, x_range, y_range, weight, m):
  corr = np.zeros((N, 1))

  for i in range(N):
    size = pos_phy[i].shape[1]
    Y = np.concatenate([pos_phy[i], np.zeros((1, size))], axis = 0)
    corr_cur = MU.mapCorrelation(m['map'], xim, yim, Y[0 : 3, :], x_range, y_range)
    ind = np.argmax(corr_cur)

    corr[i] = corr_cur[ind / 3, ind % 3]
    particles[i, 0] += x_range[ind / 3]
    particles[i, 1] += y_range[ind % 3]

  weii = np.log(weight) + corr
  weii_max = weii[np.argmax(weii)]
  lse = np.log(np.sum(np.exp(weii - weii_max)))
  weii = weii - weii_max - lse

  wei_update = np.exp(weii)
  ind_target = wei_update.argmax()

  return particles, wei_update, ind_target



'''
  the main slam section
'''
def main():
  # read in data
  joint = ld.get_joint("train_joint1")
  lid = ld.get_lidar("train_lidar1")

  # initialize variables and particles
  angles = np.array([np.arange(-135, 135.25, 0.25) * np.pi / 180.0])

  N, N_threshold = 100, 35 # the number of ideal remained particles
  particles = np.zeros((N, 3))
  weight = np.einsum('..., ...', 1.0 / N, np.ones((N, 1)))
  
  # create map
  mapp = map_init()
  
  # create global dictionary to store scan position both in physical and grid cell system
  pos_phy, posX_map, posY_map = {}, {}, {}
  
  # scale factor applied for noise
  factor = np.array([1, 1, 10])

  x_im = np.arange(mapp['xmin'], mapp['xmax'] + mapp['res'], mapp['res'])  # x-positions of each pixel of the map
  y_im = np.arange(mapp['ymin'], mapp['ymax'] + mapp['res'], mapp['res'])  # y-positions of each pixel of the map

  x_range = np.arange(-0.05, 0.06, 0.05)
  y_range = np.arange(-0.05, 0.06, 0.05)

  # plot animation map
  fig = plt.figure(1)
  ax = fig.add_subplot(111)
  ax.set_title("SLAM Map")

  im = ax.imshow(mapp['show_map'], cmap = "hot")
  fig.show()

  ts = joint['ts']
  h_angle = joint['head_angles']
  rpy_robot = joint['rpy']

  # initialize the first scan map
  lid_p = lid[0]
  rpy_p = lid_p['rpy']
  ind_0 = match_t(ts, lid_p['t'][0][0])
  pos_phy, posX_map, posY_map = scan_transform(lid_p['scan'], rpy_robot[:, ind_0], h_angle[:, ind_0], angles, particles, N, pos_phy, posX_map, posY_map, mapp)
  mapp = map_est(particles[0, :], posX_map[0], posY_map[0], mapp)

  # main SLAM Loop
  pose_p, yaw_p = lid_p['pose'], rpy_p[0, 2]
  timeline = len(lid)
  for i in xrange(1, timeline):
    lid_c = lid[i]
    pose_c, rpy_c = lid_c['pose'], lid_c['rpy']
    yaw_c = rpy_c[0, 2]

    # prediction step
    ut = control_sig(N, pose_p, pose_c, yaw_p, yaw_c, particles[:, 2])
    noise = np.einsum('..., ...', factor, np.random.normal(0, 1e-3, (N, 1)))
    particles = particles + ut + noise

    # update step
    scan_c = lid_c['scan']
    ind_i = match_t(ts, lid_c['t'][0][0])
    pos_phy, posX_map, posY_map = scan_transform(scan_c, rpy_robot[:, ind_i], h_angle[:, ind_i], angles, particles, N, pos_phy, posX_map, posY_map, mapp)
    particles, weight, ind_best = obser_model(N, pos_phy, particles, x_im, y_im, x_range, y_range, weight, mapp)

    # label the trajectories of robot
    x_r = (np.ceil((particles[ind_best, 0] - mapp['xmin']) / mapp['res']).astype(np.int16) - 1)
    y_r = (np.ceil((particles[ind_best, 1] - mapp['xmin']) / mapp['res']).astype(np.int16) - 1)
    mapp['show_map'][x_r, y_r, 0] = 255

    # use the best particle to update map
    mapp = map_est(particles[ind_best, :], posX_map[ind_best], posY_map[ind_best], mapp)

    # pass pose and yaw angle to next iteration
    pose_p, yaw_p = pose_c, yaw_c

    # resample particles if necessary
    N_eff = 1 / np.sum(np.square(weight))
    if N_eff < N_threshold:
      particles = resample(N, weight, particles)
      weight = np.einsum('..., ...', 1.0 / N, np.ones((N, 1)))

    # plot the map
    ax.imshow(mapp['show_map'], cmap = "hot")
    im.set_data(mapp['show_map'])
    im.axes.figure.canvas.draw()


  # plot the final slam map  
  fig1 = plt.figure(1)
  plt.imshow(mapp['show_map'], cmap = "hot")
  plt.show()



if __name__ == "__main__":
  main()

