import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.cluster import KMeans
import math
import pdb


'''
  save trained models
'''
def save_obj(res_dir, obj, name):
  with open(res_dir + '/'+ name + '.pkl', 'wb') as f:
    pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


'''
  load trained models
'''
def load_obj(res_dir, name):
  with open(res_dir + '/' + name + '.pkl', 'rb') as f:
    return pickle.load(f)



'''
  load training data and use kmeans to separate them
'''
def kmeansClusterData(res_dir, folder, numCluster):
  i = 1
  for fi in sorted(os.listdir(folder)):
    if fi.endswith(".txt"):
      f_name = os.path.join(folder, fi)
      arr = np.loadtxt(f_name)
      if i == 1:
        dat = arr[:, 1:]
      else:
        dat = np.vstack((dat, arr[:, 1:]))
      
      i += 1

  # K-means to cluster data
  kmeans = KMeans(n_clusters=numCluster, init='k-means++', max_iter=100).fit(dat)
  save_obj(res_dir, kmeans, 'kmeans_mod')
