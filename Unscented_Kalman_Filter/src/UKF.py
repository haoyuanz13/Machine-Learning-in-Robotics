import numpy as np 
import rotplot as rpt 
import matplotlib.pyplot as plt
import scipy.io as sio
import cv2, os
# below are helper functions 
import quaternions as quat 
import display as display
import ADC as ADC

# pick up sigma points
def sigma(state, covar, noise):
  covar = 3 * (covar + noise)

  s = np.linalg.cholesky(covar)
  s_T = s.transpose()

  W = np.zeros((6, 4))
  
  W[0:3, 1:4] = s_T
  W[3:6, 1:4] = -1 * s_T
  
  W = quat.exp_qua(W)

  sigma = quat.multi_quas(state, W)
  return sigma

# expected covariance
def cor_Py(s, s_mean):
  s_mean_inv = quat.inverse_qua(s_mean)
  Wi = quat.multi_quas(s, s_mean_inv)
  Wi_3d = quat.log_qua(Wi)[:, 1:4]

  total = Wi_3d.shape[0]

  cov = np.zeros((3, 3))
  for i in range(total):
      cov += np.outer(Wi_3d[i, :], Wi_3d[i, :])
  cov = cov / float(total)

  return [cov, Wi_3d]  

# filter 
def ukf(ut, acc):
  # initialize quaternion and covariance 
  n = 3
  # three scale factors
  # represent initial covariance, process noise and measurement noise respectfully
  scale = [1e-8, 1e-10, 1e-8]

  cor_0 = scale[0] * np.eye(n)
  noiseQ = scale[1] * np.eye(n)
  noiseR = scale[2] * np.eye(n)
  
  q0 = np.array([1, 0, 0, 0]).reshape(-1, 4)  # keep state in 4D
  period = len(ut)
  res = []
  for i in range (period):
    # pick up sigma points
    sig = sigma(q0, cor_0, noiseQ)

    # predict step
    y = quat.multi_quas(sig, ut[i])
    y_mean = quat.aver_quas_weight(y, [1 / float(2 * n)] * 6)

    xk_mins = y_mean
    [pk_mins, diff_y] = cor_Py(y, y_mean)
  
    # update step 
    g = np.array([0, 0, 0, 1]).reshape(-1, 4)
    temp = quat.multi_quas(quat.inverse_qua(y), g)
    z_acc = quat.multi_quas(temp, y)[:, 1:4]
  
    zk_mins = np.sum(z_acc, axis = 0) / float(2 * n)
  
    diff_z = z_acc - zk_mins

    pzz = np.zeros((3, 3))
    for k in range (2 * n):
      pzz += np.outer(diff_z[k, :], diff_z[k, :])
    pzz = pzz / float(2 * n)

    # other variables
    pvv = pzz + noiseR

    pxz = np.zeros((3, 3))
    for j in range(2 * n):
      pxz += np.outer(diff_y[j, :], diff_z[j, :])
    pxz = pxz / float(2 * n)

    Kal_G = np.dot(pxz, np.linalg.inv(pvv))
  
    zk_actual = acc[i].reshape(-1, n)  
    vk = zk_actual - zk_mins
  
    err = np.zeros((1, 4))
    err[:, 1:4] = np.dot(Kal_G, vk.transpose()).transpose()
    err = quat.exp_qua(err / 2)
    # update xk
    xk = quat.multi_quas(err, xk_mins)
    q0 = xk
  
    # update pk
    pk = pk_mins - np.dot(np.dot(Kal_G, pvv), Kal_G.transpose())
    cor_0 = pk

    euler_angle = quat.qua_to_ea(q0[0])# convert to Euler Angles
    res.append(euler_angle)
  return res

# main code
def main():
  folder = "imu"   # the folder stores all imu data (mat) should be in the same directory
  num_data = ADC.form_convert(folder) # convert raw data and store in suitable format

  for ind in range (num_data):
    ADC.data_convert(ind)
    q_delta = np.load('q_delta_test.npy')
    acc_actual = np.load('zk_actual_test.npy')
  
    res_filter = ukf(q_delta, acc_actual)
    # display result
    display.plot(res_filter)
    # store result
    sio.savemat('result.mat', {'f_EA':res_filter})

main()






