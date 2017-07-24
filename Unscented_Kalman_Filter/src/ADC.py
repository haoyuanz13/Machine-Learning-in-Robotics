import numpy as np
import quaternions as quat
from scipy import io
import cv2, os

# unit is g
def convert_acc(raw):
  bias = np.array([1.6467741935483873, 1.6161290322580644, 1.6516129032258065]) * 1023 / float(3.3)
  raw = raw - bias
  sensi = 300

  res = raw * 3300 / 1023 / sensi
  return res * np.array([-1, -1, 1])

# unit is degree per sec
def convert_gyro(raw):
  bias = np.array([1.1935483870967742, 1.206451612903226, 1.2129032258064516]) * 1023 / float(3.3)
  raw = raw - bias
  sensi = 3.33 * 180 / np.pi

  res = raw * 3300 / 1023 / sensi
  return res

# construct control input
def gen_control(gyro, interval):
  order_gyro = np.array([gyro[1], gyro[2], gyro[0]]).reshape(-1, 3)
  
  gyro_mod = quat.mod_qua(order_gyro)
  axis = quat.normalize_qua(order_gyro)
  
  theta = gyro_mod * interval # unit is rad per sec
  sca = np.cos(theta / float (2))
  vec = np.sin(theta / float (2)) * axis
  qua = np.zeros((1, 4))
  
  qua[:, 0] = sca
  qua[:, 1:4] = vec
  return qua

# ADC 
def data_convert(index):
  val = np.load('imu_val_test.npy')[index]
  ts = np.load('imu_ts_test.npy')[index]
  zk_actual = []
  ut = []

  total = ts.shape[0]
  for i in range (total - 1):
  	raw_acc = val[i][np.arange(3)]
  	raw_gyro = val[i][np.arange(3, 6)]

  	gyro = convert_gyro(raw_gyro)
  	t = ts[i + 1] - ts[i]

  	ut.append(gen_control(gyro, t[0]))
  	zk_actual.append(convert_acc(raw_acc))

  np.save('zk_actual_test.npy', zk_actual)
  np.save('q_delta_test.npy', ut)

# store data in suitable format for using
def form_convert(folder):
  imuu = []
  tss = []
  num = 0
  for mat in os.listdir(folder):
  	imu = io.loadmat(os.path.join(folder, mat))
  	num += 1
  	val = imu["vals"].T
  	total = np.shape(imu["ts"])[1]
  	res = {}
  	for i in range (total):
  		res[i] = val[i, :]
        imuu.append(res)
        tss.append(imu["ts"].T)

  np.save('imu_val_test.npy', imuu)
  np.save('imu_ts_test.npy', tss)
  # return the total number of imu data sets 
  return num 
