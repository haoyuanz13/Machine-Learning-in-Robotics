import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import os
import argparse
import pickle
from sklearn.cluster import KMeans
import math
import pdb

from hmm import HMM
from utils import *


################
# Define args  #
################
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_name', dest='dataset_name', default='test_multiple', help='name of the test dataset [test_multiple, test_single]')
parser.add_argument('--model_dir', dest='model_dir', default='./hmm_trained_models', help='all models are saved here')
parser.add_argument('--res_dir', dest='res_dir', default='./hmm_test_res', help='all test results are saved here')
parser.add_argument('--N', dest='N', type=int, default=15, help='the number of hidden state types')
parser.add_argument('--scal', dest='scal', type=float, default=1e-10, help='the scale factor to avoid numerical overflow')
parser.add_argument('--save_confs', dest='save_confs', type=bool, default=False, help='if save confidence images and save')
args = parser.parse_args()



'''
  main test
'''
def test():
  # def train_HMM(gest, folder, N, M):
  kmeans = load_obj(args.model_dir, 'kmeans_mod')

  # In alphabetical order
  gest = sorted(['beat3', 'beat4', 'circle', 'eight', 'inf', 'wave'])


  for fi in sorted(os.listdir(args.dataset_name)):
    f_name = os.path.join(args.dataset_name, fi)
    arr = np.loadtxt(f_name)
    test_data = arr[:, 1:]

    # cluster data
    test_data_clustered = kmeans.predict(test_data)
    t_total = len(test_data_clustered)
    

    '''
      travser all models to compute all log-likelihoods
    '''
    log_lik = []
    for model_name in gest:
      model_file = 'trained_' + model_name
      model_cur = load_obj(args.model_dir, model_file)

      T_cur, B_cur, Pi_cur = model_cur['T'], model_cur['B'], model_cur['Pi']


      '''
        Forward algo to calculate alphas
      '''
      alpha = np.zeros((t_total, args.N))

      ct = np.zeros((1, t_total))   # scale factor
      ct_raw = np.zeros((1, t_total))  # store raw factor

      # initialize base case t == 0
      alpha[0, :] = Pi_cur * B_cur[test_data_clustered[0], :]

      ct[0, 0] = 1.0 / max([args.scal, np.sum(alpha[0, :])])
      ct_raw[0, 0] = 1.0 / max([args.scal, np.sum(alpha[0, :])])
      
      alpha[0, :] = np.einsum('..., ...', ct[0, 0], alpha[0, :])  # normalize


      # run forward algorithm when t > 0
      for t in xrange(1, t_total):
        # vectorization
        alpha_pre = (alpha[t - 1, :].reshape(-1, args.N)).T
        alpha[t, :] = (B_cur[test_data_clustered[t], :]) * (np.einsum('ij,jk->ik', T_cur, alpha_pre)).T

        ct[0, t] = 1.0 / max(args.scal, np.sum(alpha[t, :]))
        ct_raw[0, t] = 1.0 / max(args.scal, np.sum(alpha[t, :]))
        
        # normalize
        alpha[t, :] = np.einsum('..., ...', ct_raw[0 ,t], alpha[t, :])
      

      # get log-likelihood among all gesture classes
      log_prob = -np.sum(np.log(ct_raw))

      if math.isnan(log_prob):
        log_lik.append(-float("inf"))
      else:
        log_lik.append(log_prob)


    '''
      obtain the index with highest log-likelihood
    '''
    idx = log_lik.index(max(log_lik))
    conf = (1. / np.asarray(log_lik)) / np.sum(1. / np.asarray(log_lik))
  
    # print conf    
    print 'The test file [' + fi[0 : -4] + '] is predicted as the gesture [' + gest[idx] + '] with confidence {:.4f}%.'.format(conf[idx] * 100)


    if args.save_confs:
      x = np.arange(len(gest))
      plt.bar(x, height=list(conf))
      plt.xticks(x + .5, gest)
      plt.xlabel('Gestures')
      plt.ylabel('Confidence')
      s = args.res_dir + '/' + args.dataset_name + '_' + fi[0 : -4]
      plt.savefig(s, bbox_inches='tight')
      plt.clf()



if __name__ == "__main__":
  # check necessary directories
  print ("\n=====>> Checking necessary directories ..........")
  if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir)
  
  test()
