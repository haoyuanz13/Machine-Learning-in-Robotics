import numpy as np
from hmm import HMM
import readinData as rd
import os

'''
  define global variables
'''
N, M = 15, 40
T = np.random.uniform(0, 1, (N, N))
T /= np.sum(T, axis = 0)

B = np.random.uniform(0, 1, (M, N))
B /= np.sum(B, axis = 0)

Pi = np.random.uniform(0, 1, (1, N))
Pi /= np.sum(Pi)


'''
  main training step
'''
def main():
  folder = 'inf'   # the folder stores data you want to train
  obs_set, num_data = {}, 0
  for txt in os.listdir(folder):
    ts, obs = rd.disData(folder, txt, M)
    obs_set[num_data] = obs.astype(np.int16)
    num_data += 1

  iter, threshold = 1000, 1e-4
  gam_set, sigma_set, Pi_set = {}, {}, {}
  hmm = HMM(T, B, Pi[0], N, M)
  log_prob_pre = 0

  for i in range(iter):
    # EM step uses all data
    for ind_data in range(num_data):
      obs = obs_set[ind_data]
      Pi_cur, gam_cur, sigma_cur = hmm.E_Step(obs)

      gam_set[ind_data] = gam_cur
      sigma_set[ind_data] = sigma_cur
      Pi_set[ind_data] = Pi_cur

    hmm.M_Step(num_data, obs_set, gam_set, sigma_set, Pi_set)

    # compute the log prob with estimated parameters
    sum_prob = 0
    for j in range(num_data):
      obs = obs_set[j]
      sum_prob += hmm.forward(obs)[0]
    log_prob = sum_prob / float(num_data)
    
    if np.abs(log_prob - log_prob_pre) < threshold:
      break
    log_prob_pre = log_prob

    # print np.sum(hmm.T, axis=0)
    # print np.sum(hmm.B, axis=0)
    # print np.sum(hmm.Pi)
  
  res = {'T': hmm.T, 'B': hmm.B, 'Pi': hmm.Pi}
  np.save('%s.npy'%folder, res)


if __name__ == '__main__':
  main();
  



