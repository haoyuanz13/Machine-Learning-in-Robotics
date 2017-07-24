import load_data as ld
import numpy as np
import matplotlib.pyplot as plt
import MapUtils as MU
import cv2
import random


# initialize map
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


# time stamp match
def match_t(t_head, t_odo):
    t_temp = t_head - t_odo
    error = np.absolute(t_temp)
    index = np.argmin(error)
    return index


# transfer frame
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


# transfer scan data into the global frame
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


# estimate map
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


# motion model: compute control signal based on lidar pose
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


# observation model: recompute weight
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


# resample step to uniform weight
def resample(N, weight, particles):
    particle_New = np.zeros((N, 3))
    r = random.uniform(0, 1.0 / N)

    c = weight[0]
    i = 0
    for m in range(N):
        u = r + m * (1.0 / N)
        while u > c:
            i = i + 1
            c = c + weight[i]

        particle_New[m, :] = particles[i, :]
    return particle_New


# main code
def main():
    # read in data
    joint = ld.get_joint("train_joint1")
    lid = ld.get_lidar("train_lidar1")

    # initialize variables
    angles = np.array([np.arange(-135, 135.25, 0.25) * np.pi / 180.0])

    N, N_threshold = 100, 35 # the number of ideal remained particles
    particles = np.zeros((N, 3))
    weight = np.einsum('..., ...', 1.0 / N, np.ones((N, 1)))
    # create map
    mapp = map_init()
    # create global dictionary to store scan position both in physical and map coordinate
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
    total = len(lid)

    # initialize the first scan map
    lid_p = lid[0]
    rpy_p = lid_p['rpy']
    ind_0 = match_t(ts, lid_p['t'][0][0])
    pos_phy, posX_map, posY_map = scan_transform(lid_p['scan'], rpy_robot[:, ind_0], h_angle[:, ind_0], angles, particles, N, pos_phy, posX_map, posY_map, mapp)
    mapp = map_est(particles[0, :], posX_map[0], posY_map[0], mapp)

    # SLAM Loop
    pose_p = lid_p['pose']
    yaw_p = rpy_p[0, 2]
    for i in range(1, total):
        lid_c = lid[i]
        pose_c = lid_c['pose']
        rpy_c = lid_c['rpy']
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
        pose_p = pose_c
        yaw_p = yaw_c

        # resample particles if necessary
        N_eff = 1 / np.sum(np.square(weight))
        if N_eff < N_threshold:
            particles = resample(N, weight, particles)
            weight = np.einsum('..., ...', 1.0 / N, np.ones((N, 1)))

        # plot the map
        ax.imshow(mapp['show_map'], cmap = "hot")
        im.set_data(mapp['show_map'])
        im.axes.figure.canvas.draw()

    fig1 = plt.figure(1)
    plt.imshow(mapp['show_map'], cmap = "hot")
    plt.show()

main()

