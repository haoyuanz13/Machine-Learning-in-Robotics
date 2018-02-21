import numpy as np
import cv2
import random


'''
  Convert Euler angle to rotation matrix
'''
def ea2rot(r, p, y):
  r11 = np.cos(y) * np.cos(p)
  r12 = np.cos(y) * np.sin(p) * np.sin(r) - np.sin(y) * np.cos(r)
  r13 = np.cos(y) * np.sin(p) * np.cos(r) + np.sin(y) * np.sin(r)

  r21 = np.sin(y) * np.cos(p)
  r22 = np.sin(y) * np.sin(p) * np.sin(r) + np.cos(y) * np.cos(r)
  r23 = np.sin(y) * np.sin(p) * np.cos(r) - np.cos(y) * np.sin(r)

  r31 = -np.sin(p)
  r32 = np.cos(p) * np.sin(r)
  r33 = np.cos(p) * np.cos(r)

  R = np.array([[r11, r12, r13],
                [r21, r22, r23],
                [r31, r32, r33]])
  return R


'''
  initialize map: 2D occupied grid cell
'''
def map_init():
  MAP = {}
  MAP['res'] = 0.05  # meters
  MAP['xmin'] = -40  # meters
  MAP['ymin'] = -40
  MAP['xmax'] = 40
  MAP['ymax'] = 40
  MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1))  # cells
  MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))

  MAP['log_map'] = np.zeros((MAP['sizex'], MAP['sizey'])) # log map
  MAP['map'] = np.zeros((MAP['sizex'], MAP['sizey']), dtype = np.int8)  # DATA TYPE: char or int8
  MAP['show_map'] = 0.5 * np.ones((MAP['sizex'], MAP['sizey'], 3), dtype = np.int8)  # DATA TYPE: char or int8
  
  return MAP


'''
  extract timeline
'''
def time_extract(dep):
  dep_size = len(dep)
  t_dep = np.zeros((1, dep_size))
  
  for i in range(dep_size):
    dep_cur = dep[i]
    t_dep[0, i] = dep_cur['t'][0][0]
  
  return t_dep

  
'''
  time stamp match
'''
def match_t(t_head, t_odo):
  t_temp = t_head - t_odo
  error = np.absolute(t_temp)
  index = np.argmin(error)
  return index


'''
  frame transformation
'''
def trans_frame(part_cur, ori_robot, head_angles):
  r, p, y = ori_robot[0], ori_robot[1], part_cur[2]
  r11 = np.cos(y) * np.cos(p)
  r12 = np.cos(y) * np.sin(p) * np.sin(r) - np.sin(y) * np.cos(r)
  r13 = np.cos(y) * np.sin(p) * np.cos(r) + np.sin(y) * np.sin(r)

  r21 = np.sin(y) * np.cos(p)
  r22 = np.sin(y) * np.sin(p) * np.sin(r) + np.cos(y) * np.cos(r)
  r23 = np.sin(y) * np.sin(p) * np.cos(r) - np.cos(y) * np.sin(r)

  r31 = -np.sin(p)
  r32 = np.cos(p) * np.sin(r)
  r33 = np.cos(p) * np.cos(r)
  
  # transfer from world to body
  t_w2b = np.array([[r11, r12, r13, part_cur[0]],
                    [r21, r22, r23, part_cur[1]],
                    [r31, r32, r33, 0.93],
                    [0, 0, 0, 1]])
  
  # transfer from body to head
  t_b2h = np.array([[np.cos(head_angles[0]), -np.sin(head_angles[0]), 0, 0],
                    [np.sin(head_angles[0]), np.cos(head_angles[0]), 0, 0],
                    [0, 0, 1, 0.33],
                    [0, 0, 0, 1]])
  
  # transfer from head to lidar
  t_h2l = np.array([[np.cos(head_angles[1]), 0, np.sin(head_angles[1]), 0],
                    [0, 1, 0, 0],
                    [-np.sin(head_angles[1]), 0, np.cos(head_angles[1]), 0.15],
                    [0, 0, 0, 1]])

  T = np.einsum('ij,jk,kl->il', t_w2b, t_b2h, t_h2l)
  return T


'''
  transfer the scan data into the global frame
  - Input N: the number of particles
  - Input m: estimated map
'''
def scan_transform(scan, ori_robot, head_a, angles, particles, N, pos_phy, posX_map, posY_map, m):
  # remove scan data too close or too far
  indValid = np.logical_and((scan < 30), (scan > 0.1))
  scan_valid = scan[indValid]
  angles_valid = angles[indValid]

  xs0 = np.array([np.einsum('i,i->i', scan_valid, np.cos(angles_valid))])
  ys0 = np.array([np.einsum('i,i->i', scan_valid, np.sin(angles_valid))])

  Y = np.concatenate([np.concatenate([np.concatenate([xs0, ys0], axis=0), np.zeros(xs0.shape)], axis=0), np.ones(xs0.shape)], axis=0)
  
  for i in range(N):
    # trans_cur = trans.trans_frame(particles[i, :], ori_robot, head_a)
    trans_cur = trans_frame(particles[i, :], ori_robot, head_a)

    res = np.einsum('ij,jk->ik', trans_cur, Y)
    ind_notG = res[2, :] > 0.1   # remove ground points

    pos_phy[i] = res[0 : 2, ind_notG]
    posX_map[i] = (np.ceil((res[0, ind_notG] - m['xmin']) / m['res']).astype(np.int16) - 1)
    posY_map[i] = (np.ceil((res[1, ind_notG] - m['ymin']) / m['res']).astype(np.int16) - 1)

  return pos_phy, posX_map, posY_map


'''
 Use the best particle to estimate and update map
'''
def map_est(particle_cur, xis, yis, m):
  x_sensor = (np.ceil((particle_cur[0] - m['xmin']) / m['res']).astype(np.int16) - 1)
  y_sensor = (np.ceil((particle_cur[1] - m['ymin']) / m['res']).astype(np.int16) - 1)

  x_occupied = np.concatenate([xis, [x_sensor]])
  y_occupied = np.concatenate([yis, [y_sensor]])

  m['log_map'][xis, yis] += 2 * np.log(9)
  polygon = np.zeros((m['sizey'], m['sizex']))

  occupied_ind = np.vstack((y_occupied, x_occupied)).T
  cv2.drawContours(image = polygon, contours = [occupied_ind], contourIdx = 0, color = np.log(1.0 / 9), thickness = -1)
  m['log_map'] += polygon

  occupied = m['log_map'] > 0
  empty = m['log_map'] < 0
  route = (m['show_map'][:, :, 0] == 255)

  m['map'][occupied] = 1
  m['show_map'][occupied, :] = 0
  m['show_map'][np.logical_and(empty, ~route), :] = 1

  return m


'''
  resample particles to make them uniform and diverse
'''
def resample(N, weight, particles):
  particle_New = np.zeros((N, 3))
  r = random.uniform(0, 1.0 / N)

  c, i = weight[0], 0
  for m in range(N):
    u = r + m * (1.0 / N)
    
    while u > c:
      i = i + 1
      c = c + weight[i]

    particle_New[m, :] = particles[i, :]
  
  return particle_New
