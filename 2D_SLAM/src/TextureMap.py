import numpy as np
import pickle
import load_data as ld


# EA angle to rotation matrix
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

# extract timeline
def time_extract(dep):
    dep_size = len(dep)
    t_dep = np.zeros((1, dep_size))
    for i in range(dep_size):
        dep_cur = dep[i]
        t_dep[0, i] = dep_cur['t'][0][0]
    return t_dep


# match time stamp
def time_match(timeline, t_cur):
    t_temp = timeline - t_cur
    error = np.absolute(t_temp)
    index = np.argmin(error)

    # return index if the time error is below than certain threshold
    return index if error[index] < 0.1 else -1


# estimate coefficient value
def coff_est(dep_cur):
    rpy = dep_cur['imu_rpy']
    R = ea2rot(rpy[0, 0], rpy[0, 1], rpy[0, 2])
    A = np.einsum('ij,jk->ik', R, np.array([[0], [1], [0]]))
    a0, a1, a2 = A[0, 0], A[1, 0], A[2, 0]
    return a0, a1, a2


# find ground plane
def find_ground(a0, a1, a2, a3, dep_cur):
    depz = dep_cur['depth'][u_dep, v_dep]
    uu = u_dep - cu_dep
    vv = v_dep - cv_dep

    t1 = a0 * (uu * depz) / fu_dep
    t2 = a1 * (vv * depz) / fv_dep
    t3 = a2 * depz + a3
    val = np.absolute(t1 + t2 + t3)

    u_gnd = u_dep[val < 0.1]
    v_gnd = v_dep[val < 0.1]
    return u_gnd, v_gnd


# transfer depth camera to rgb camera
def trans_dep2rgb(u_gnd, v_gnd, dep_cur, homo):
    depz = dep_cur['depth'][u_gnd, v_gnd]
    xir = (u_gnd - cu_dep) * depz / fu_dep
    yir = (v_gnd - cv_dep) * depz / fv_dep

    total = len(xir)
    Y = np.concatenate([np.concatenate([np.concatenate([[xir], [yir]], axis=0), np.ones((1, total))], axis=0), np.ones((1, total))], axis=0)
    X_rgb = np.einsum('ij,jk->ik', homo, Y)

    uu_rgb = fu_rgb * X_rgb[0, :] / X_rgb[2, :]
    vv_rgb = fv_rgb * X_rgb[1, :] / X_rgb[2, :]

    return uu_rgb, vv_rgb, X_rgb


def main(time_stamp):
    dep = ld.get_depth("DEPTH_0")
    rgb = ld.get_rgb("RGB_0")
    time_line = time_extract(dep)
    t = time_match(time_line, time_stamp)

    # when find the match time stamp
    if t != -1:
        global fu_rgb, fv_rgb, cu_rgb, cv_rgb
        fu_rgb, fv_rgb = 1049.3317526, 1051.31847629
        cu_rgb, cv_rgb = 956.910516428, 533.452032441

        global fu_dep, fv_dep, cu_dep, cv_dep
        fu_dep, fv_dep = 364.457362486, 364.542810627
        cu_dep, cv_dep = 258.422487562, 202.48713994

        global u_dep, v_dep
        v_dep, u_dep = np.meshgrid(np.arange(0, 512, 1), np.arange(0, 424, 1))
        u_dep = u_dep.reshape(217088, 1)
        v_dep = v_dep.reshape(217088, 1)

        dep_cur = dep[t]
        rgb_cur = rgb[t]
        im = rgb_cur['image']
        # compute the coefficient a0, a1 and a2
        a0, a1, a2 = coff_est(dep_cur)
        a3 = -0.93 - 0.33 - 0.07
        # find pixels belongs to ground plane
        # the return result is the pixel position in the depth image without normalizing
        u_gnd_dep, v_gnd_dep = find_ground(a0, a1, a2, a3, dep_cur)
        # homography
        homo_dep2rgb = np.array([[0.99996855, 0.00589981, 0.00529992, 52.2682],
                                [-0.00589406, 0.99998202, -0.00109998, 1.5192],
                                [-0.00530632, 0.00106871, 0.99998535, -0.6059],
                                [0, 0, 0, 1]])
        # transfer to rgb camera
        u_gnd_rgb, v_gnd_rgb, pos_IR = trans_dep2rgb(u_gnd_dep, v_gnd_dep, dep_cur, homo_dep2rgb)
        u_gnd_rgb = round(u_gnd_rgb + cu_rgb)
        v_gnd_rgb = round(v_gnd_rgb + cv_rgb)
        # rgb value at (u, v) position
        val_rgb = im[u_gnd_rgb, v_gnd_rgb, :]
        return val_rgb, pos_IR  # pos_IR is the physical coordinate in 3D space seen by IR camera

    # means no time stamp match
    else:
        return [], []

# test code
time_stamp = 120909443.89  # input time stamp
main(time_stamp)