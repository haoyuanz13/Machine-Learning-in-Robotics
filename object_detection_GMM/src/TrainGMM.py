from __future__ import division
import numpy as np
import cv2, os
import math
import random as random
import copy

barr = np.load('data_redBarrel.npy') # read data in
barrel = np.array(barr, dtype=int)

max_col = barrel.max(axis=0)
min_col = barrel.min(axis=0)
diff = (max_col - min_col) / float(2)

size = len(barrel)

# Applying EM to train GMM
k = 4
set_GMM = []

# initialize
for i in range (k):
  table = {}
  table["r"] =  np.zeros((1, size))
  table["mean"] = barrel[random.randint(0, size - 1), :]
  table["covar"] = np.array([[diff[0], 0, 0], [0, diff[1], 0], [0, 0, diff[2]]])
  table["wei"] = 1 / float(k)
  set_GMM.append(table)


'''
  Compute probability for each training data
'''
def set_of_pdf(mean, var, wei):
  det_var = np.linalg.det(var) # det value of variance
  inv_var = np.linalg.inv(var) # inverse value of variance

  # Gaussian model 
  term_a = (-1.5) * math.log(2 * math.pi)
  term_b = (-0.5) * math.log(det_var)

  norm = np.matrix(barrel - mean)
  x = np.sum(np.multiply(norm * inv_var, norm), axis = 1) / float(-2)
  prob = term_a + term_b + x

  res = np.multiply(np.exp(prob), wei)
  return res.transpose()

'''
  EM interation 
'''
ite, pre = 500, 0
while (ite >= 0):
  # E step
  member_prob = np.zeros((k, size))
  for i in range (k):
    cur_mean = set_GMM[i]["mean"]
    cur_var = set_GMM[i]["covar"]
    cur_weight = set_GMM[i]["wei"]
    
    member_prob[i, :] = set_of_pdf(cur_mean, cur_var, cur_weight)
      
  deno = np.matrix(np.sum(member_prob, axis = 0))

  # detect the difference bwteen current iteration and previous one
  max_likehood = np.sum(deno, axis = 1)[0, 0]
  print max_likehood
  if math.fabs(max_likehood - pre) < 1e-6:
    break;
  pre = max_likehood

  for i in range (k):
    cur_r = set_GMM[i]["r"]
    set_GMM[i]["r"] = (member_prob[i, :] / deno).transpose()

  # M step
  for i in range (k):
    cur_r = set_GMM[i]["r"]
    sum_r = np.sum(cur_r, axis = 0)
    
    # avoid overflow problem
    if sum_r < 1.7e-308:
      sum_r = 1.7e-308
        
    # update mean
    temp_mean = np.multiply(cur_r, barrel).sum(axis = 0)
    set_GMM[i]["mean"] = temp_mean / float(sum_r)

    # update variance 
    cur_mean = set_GMM[i]["mean"]
    temp_var = np.zeros((3, 3))
    for x in range (size):
      norm = barrel[x, :] - cur_mean
      temp_var += np.multiply(cur_r[x, :], np.outer(norm, norm))
    
    set_GMM[i]["covar"] = temp_var / float(sum_r)

    # update weight
    set_GMM[i]["wei"] = sum_r / float (size)
  
  ite -= 1

# save trained model
np.save('GMM_red_Update.npy', set_GMM)






