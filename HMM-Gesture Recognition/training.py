import numpy as np
import os
import numpy.matlib
import matplotlib.pyplot as plt
import os
import argparse
import pickle
from sklearn.cluster import KMeans
import math
import pdb

from hmm import *
from utils import *


################
# Define args  #
################
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_name', dest='dataset_name', default='train', help='name of the training dataset')
parser.add_argument('--res_dir', dest='res_dir', default='./hmm_trained_models', help='all models are saved here')
parser.add_argument('--N', dest='N', type=int, default=15, help='the number of hidden state types')
parser.add_argument('--M', dest='M', type=int, default=35, help='the number of observation state types')
parser.add_argument('--scal', dest='scal', type=float, default=1e-10, help='the scale factor to avoid numerical overflow')
parser.add_argument('--iter_max', dest='iter_max', type=int, default=500, help='# of training epoch')
parser.add_argument('--eps', dest='eps', type=float, default=1e-7, help='the threashold to determine whether training is completed')
parser.add_argument('--plot_cost', dest='plot_cost', type=bool, default=False, help='if plot training log-likelihood')
parser.add_argument('--needCluster', dest='needCluster', type=bool, default=False, help='if use k-means to cluster data')
args = parser.parse_args()



'''
  main training step
'''
def train_HMM(ges_cur, obs_set, N, M, iter_max):
  # overall batch size
  batch_size = len(obs_set)

  # build a new model
  hmm = HMM(args.scal, N, M)
  
  log_prob_pre = float("inf")
  lop_prob_list = []

  # training iteration
  for i in xrange(iter_max):
    gam_set, sigma_set, Pi_set = {}, {}, {}
    
    # E-step: includes FP and BP to get gam, sigma and Pi set
    log_prob_sum = 0
    for ind_batch in xrange(batch_size):
      obs = obs_set[ind_batch]
      Pi_cur, gam_cur, sigma_cur, log_prob_cur = hmm.E_Step(obs)

      gam_set[ind_batch] = gam_cur
      sigma_set[ind_batch] = sigma_cur
      Pi_set[ind_batch] = Pi_cur

      log_prob_sum += log_prob_cur

    # M-step: update model
    hmm.M_Step(batch_size, obs_set, gam_set, sigma_set, Pi_set)


    # compute the log prob with estimated parameters
    log_prob = log_prob_sum / float(batch_size)

    if i > 0:
      print '---> The log-likelihood of {:04d}th iter is: {:.6f}; The difference with {:04d}th iter is: {:.6f}.'.format(
                                i + 1, log_prob, i, log_prob - log_prob_pre)
    else:
      print '---> The log-likelihood of {:04d}th iter is: {:.6f}.'.format(i + 1, log_prob)

    lop_prob_list.append(log_prob)

    # check the stop criterion
    if math.fabs(log_prob - log_prob_pre) < args.eps:
      break
    else:
      log_prob_pre = log_prob



  print '[*] Completed training ! \n'
  
  '''
    save corresponding model
  '''
  modle_cur = {'T': hmm.T, 'B': hmm.B, 'Pi': hmm.Pi}
  model_name = 'trained_' + ges_cur

  save_obj(args.res_dir, modle_cur, model_name)

  return lop_prob_list


'''
  main training process
'''
def main(gest_list):
  # kmeans to cluster data
  if args.needCluster:
    print '[*] K-means to cluster dataset .......... \n'
    kmeansClusterData(args.res_dir, args.dataset_name, args.M)

  # load clustered data
  kmeans = load_obj(args.res_dir, 'kmeans_mod')
  clusted_data = kmeans.labels_

  cum_numData = 0
  # traverse all gesture types
  for g in gest_list:
    print '[*] Start Training: the current training gesture type is [' + g + ']. \n'


    '''
      loading all training data of the cureent gesture
    '''
    cur_ges_trainSet = []
    for fi in sorted(os.listdir(args.dataset_name)):
      if fi.startswith(g):
        f_name = os.path.join(args.dataset_name, fi)
        num_data = np.loadtxt(f_name).shape[0]
        
        sub_dataset = clusted_data[cum_numData: (cum_numData + num_data - 1)]
        cur_ges_trainSet.append(sub_dataset)

        cum_numData += num_data
    

    '''
      training HMM
    '''
    lop_probs = train_HMM(g, cur_ges_trainSet, args.N, args.M, args.iter_max)


    '''
      plot log-likelihood if necessary
    '''
    if args.plot_cost:
      plt.plot(lop_probs)
      plt.ylabel('Log Likelihood')
      plt.show()





if __name__ == '__main__':
  # check necessary directories
  print ("\n=====>> Checking necessary directories .......... \n")
  if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir)

  # gesture list for training
  gest = ['beat3', 'circle', 'beat4', 'eight', 'inf', 'wave']
  gest = sorted(gest)
  
  main(gest)
