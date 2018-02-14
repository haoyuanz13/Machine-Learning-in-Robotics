import numpy as np

'''
  HMM Model 
'''
class HMM(object):
  def __init__(self, T_ini, B_ini, Pi_ini, N, M):
    # N is the total number of hidden state, M is the total number of observation state
    # T: transition model. B: observation model. Pi: initial distribution (hidden states)
    self.numH = N
    self.numO = M

    self.T = T_ini  # N by N
    self.B = B_ini  # M by N
    self.Pi = Pi_ini # 1 by N
    return


  '''
    Forward Propagation from 0 to T
  '''
  def forward(self, obs):
    t_total = len(obs)
    epi = 1e-200
    alpha = np.zeros((t_total, self.numH))
    ct = np.zeros((1, t_total))   # scale factor
    # initialize base case t == 0
    alpha[0, :] = self.Pi * self.B[obs[0], :]

    ct_cur = 1.0 / max(epi, np.sum(alpha[0, :]))
    ct[0, 0] = ct_cur
    alpha[0, :] = np.einsum('..., ...', ct_cur, alpha[0, :])

    # run forward algorithm when t > 0
    for t in range(1, t_total):
      # vectorization
      alpha_pre = (alpha[t - 1, :].reshape(-1, self.numH)).T
      alpha[t, :] = (self.B[obs[t], :]) * (np.einsum('ij,jk->ik', self.T, alpha_pre)).T

      # for k in range(self.numH):
      #     self.alpha[t, k] = sum((self.T[kk, k] * self.B[obs[t], k] * self.alpha[t - 1, kk]) for kk in range(self.numH))

      ct_cur = 1.0 / max(epi, np.sum(alpha[t, :]))
      ct[0, t] = ct_cur
      alpha[t, :] = np.einsum('..., ...', ct_cur, alpha[t, :])
    
    log_prob = -np.sum(np.log(ct))
    return log_prob, alpha, ct


  '''
    Backward Propagation from T-1 to 0
  '''
  def backward(self, obs, ct):
    t_total = len(obs)
    beta = np.zeros((t_total, self.numH))

    # initialize base case when t == t_total
    beta[t_total- 1, :] = 1 * ct[0, t_total - 1]
    # run backward algorithm
    for t in reversed(range(t_total - 1)):
      # vectorization
      beta_fu = (beta[t + 1, :].reshape(-1, self.numH)).T
      beta[t, :] = np.dot(self.B[obs[t + 1], :], np.einsum('..., ...', self.T, beta_fu))
      beta[t, :] *= ct[0, t]
      # for k in range(self.numH):
      #     self.beta[t, k] = sum((self.T[k, kk] * self.B[kk, obs[t + 1]] * self.beta[t + 1, kk]) for kk in range(self.numH))

    # prob = sum(np.einsum('i,i,i->i', self.Pi, self.B[obs[0], :], self.beta[0, :]))
    return beta

  '''
    Baum-Welch Algorithm using EM steps
  '''
  # E step contains FP and BP to estimate alpha, beta, marginals and pair states
  def E_Step(self, obs):
    t_total = len(obs)
    gam = np.zeros((t_total, self.numH))
    sigma = np.zeros((t_total, self.numH, self.numH))
    cur_Pi = np.zeros([1, self.numH])

    log_p, alpha, ct = self.forward(obs) # estimate alpha
    beta = self.backward(obs, ct) # estimate beta

    # E step to estimate gama and sigma
    for t in range(t_total):
      nom = np.einsum('i,i->i', alpha[t, :], beta[t, :])
      gam[t, :] = nom / np.einsum('i...->...', nom)
      if t == 0:
        cur_Pi = gam[0, :]

      if t == t_total - 1:
        continue

      term1 = np.einsum('..., ...', self.T, alpha[t, :])
      term2 = np.einsum('..., ...', self.B[obs[t + 1], :], beta[t + 1, :]).reshape(-1, self.numH)
      sigma[t, :, :] = np.einsum('..., ...', term1, term2.T)
      # for k in range(self.numH):
      #     sigma[t, k, :] = np.einsum('i,i,i,i->i', self.alpha[t, :], self.T[k, :], self.B[obs[t + 1], :], self.beta[t + 1, :])
      sigma[t, :, :] /= np.sum(sigma[t, :, :])

    return cur_Pi, gam, sigma

  # M step to update model T, B and Pi
  def M_Step(self, K, obs_set, gam_set, sig_set, Pi_set):
    # update Pi
    Pis = Pi_set.values()
    self.Pi = np.sum(Pis, axis = 0) * (1.0 / K)
    # update T
    sum_sig = np.zeros((self.numH, self.numH))
    sum_gam = np.zeros((1, self.numH))

    for ind_k in range(K):
      sum_sig += np.sum(sig_set[ind_k][:-1, :, :], axis = 0)
      sum_gam += np.sum(gam_set[ind_k][:-1, :], axis = 0)

    self.T = sum_sig / sum_gam
    
    # update B
    for i in range(self.numO):
      sum_nome = np.zeros((1, self.numH))
      sum_de = np.zeros((1, self.numH))

      for ind_kk in range(K):
        obs = obs_set[ind_kk]
        gam_cur = gam_set[ind_kk]
        ind = (obs == i)

        sum_nome += np.sum(gam_cur[ind.T[0], :], axis = 0)
        sum_de += np.sum(gam_cur, axis = 0)

      self.B[i, :] = sum_nome / sum_de





















