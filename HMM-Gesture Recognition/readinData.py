import numpy as np
import cv2
import os

'''
  read in data
'''
def readIn(foldername, filename):
  filein = np.loadtxt(os.path.join(foldername, filename))
  time_line = filein[:, 0]
  val = filein[:, 1 : 7]
  return time_line, val

'''
  discrete data
'''
def disData(folder, filename, k):
  timeline, data = readIn(folder, filename)
  data = np.vstack(data).astype(np.float32)
  total = data.shape[0]

  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
  ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

  dis_data = np.zeros((total, 1))

  for i in range(k):
    index = (label.ravel() == i)
    dis_data[index] = i

  return timeline, dis_data