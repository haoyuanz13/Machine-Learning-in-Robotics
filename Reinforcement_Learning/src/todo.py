import numpy as np
from util import softmax_prob, Message, discount, fmt_row
from frozen_lake import rollout
import pdb
import scipy.signal, time

def value_iteration(env, gamma, nIt):
  """
  Inputs:
      env: Environment description
      gamma: discount factor
      nIt: number of iterations
  Outputs:
      (value_functions, policies)
      
  len(value_functions) == nIt+1 and len(policies) == nIt+1
  """
  Vs = [np.zeros(env.nS)]
  pis = [np.zeros(env.nS, dtype = 'int')]
  for it in range(nIt):
    V, pi = vstar_backup(Vs[-1], env, gamma)
    Vs.append(V)
    pis.append(pi)

  return Vs, pis

def policy_iteration(env, gamma, nIt):
  """
  Inputs:
      env: Environment description
      gamma: discount factor
      nIt: number of iterations
  Outputs:
      (value_functions, policies)
      
  len(value_functions) == nIt+1 and len(policies) == nIt+1
  """
  Vs = [np.zeros(env.nS)]
  pis = [np.zeros(env.nS,dtype='int')] 
  for it in range(nIt):
    vpi = policy_evaluation_v(pis[-1], env, gamma)
    qpi = policy_evaluation_q(vpi, env, gamma)
    pi = qpi.argmax(axis=1)
    Vs.append(vpi)
    pis.append(pi)

  return Vs, pis


def policy_gradient_optimize(env, policy, gamma,
      max_pathlength, timesteps_per_batch, n_iter, stepsize):
  from collections import defaultdict
  stat2timeseries = defaultdict(list)
  widths = (17,10,10,10,10)
  print (fmt_row(widths, ["EpRewMean","EpLenMean","Perplexity","KLOldNew"]))
  for i in range(n_iter):
      # collect rollouts
      total_ts = 0
      paths = [] 
      while True:
          path = rollout(env, policy, max_pathlength)                
          paths.append(path)
          total_ts += path["rewards"].shape[0] # Number of timesteps in the path
          #pathlength(path)
          if total_ts > timesteps_per_batch: 
              break

      # get observations:
      obs_no = np.concatenate([path["observations"] for path in paths])
      # Update policy
      policy_gradient_step(policy, paths, gamma, stepsize)

      # Compute performance statistics
      pdists = np.concatenate([path["pdists"] for path in paths])
      kl = policy.compute_kl(pdists, policy.compute_pdists(obs_no)).mean()
      perplexity = np.exp(policy.compute_entropy(pdists).mean())

      stats = {  "EpRewMean" : np.mean([path["rewards"].sum() for path in paths]),
                 "EpRewSEM" : np.std([path["rewards"].sum() for path in paths])/np.sqrt(len(paths)),
                 "EpLenMean" : np.mean([path["rewards"].shape[0] for path in paths]), #pathlength(path) 
                 "Perplexity" : perplexity,
                 "KLOldNew" : kl }
      print (fmt_row(widths, ['%.3f+-%.3f'%(stats["EpRewMean"], stats['EpRewSEM']), stats['EpLenMean'], stats['Perplexity'], stats['KLOldNew']]))
      
      for (name,val) in stats.items():
          stat2timeseries[name].append(val)
  return stat2timeseries

def vstar_backup(v_n, env, gamma):
  """
  Apply Bellman backup operator V -> T[V], i.e., perform one step of value iteration

  :param v_n: the state-value function (1D array) for the previous iteration
  :param env: environment description providing the transition and reward functions
  :param gamma: the discount factor (scalar)
  :return: a pair (v_p, a_p), where 
  :  v_p is the updated state-value function and should be a 1D array (S -> R),
  :  a_p is the updated (deterministic) policy, which should also be a 1D array (S -> A)
  """
  v_p, a_p = np.zeros(env.nS), np.zeros(env.nS)
  # update all states
  for s in range(env.nS):
      val_max = -(1e-300)
      pi_max = 0
      for a in range(env.nA):
          PR_sa = env.P[s][a]
          r_sa, pV_sa = 0, 0

          for possi in PR_sa:
              r_sa += possi[0] * possi[2]
              pV_sa += possi[0] * v_n[possi[1]]

          Q_cur = r_sa + gamma * pV_sa

          if Q_cur > val_max:
              val_max = Q_cur
              pi_max = a

      v_p[s] = val_max
      a_p[s] = pi_max

  assert v_p.shape == (env.nS,)
  assert a_p.shape == (env.nS,)  
  return (v_p, a_p)

def policy_evaluation_v(pi, env, gamma):
  """
  :param pi: a deterministic policy (1D array: S -> A)
  :param env: environment description providing the transition and reward functions
  :param gamma: the discount factor (scalar)
  :return: vpi, the state-value function for the policy pi
  
  Hint: use np.linalg.solve
  """
  P, R = np.zeros((env.nS, env.nS)), np.zeros((env.nS, 1))
  # vectorization
  for i in range(env.nS):
      a_cur = pi[i]
      P_cur = env.P[i][a_cur]

      rew_sa = 0
      for possi in P_cur:
          P[i][possi[1]] = possi[0]
          rew_sa += possi[0] * possi[2]
      R[i] = rew_sa

  A = np.eye(env.nS, env.nS) - np.einsum('..., ...', gamma, P)
  vpi = np.linalg.solve(A, R).reshape(env.nS, )
  assert vpi.shape == (env.nS,)
  return vpi

def policy_evaluation_q(vpi, env, gamma):
  """
  :param vpi: the state-value function for the policy pi
  :param env: environment description providing the transition and reward functions
  :param gamma: the discount factor (scalar)
  :return: qpi, the state-action-value function for the policy pi
  """
  qpi = np.zeros((env.nS, env.nA))

  for s in range(env.nS):
      for a in range(env.nA):
          PR_sa = env.P[s][a]
          r_sa, pV_sa = 0, 0

          for possi in PR_sa:
              pV_sa += possi[0] * vpi[possi[1]]
              r_sa += possi[0] * possi[2]

          qpi[s, a] = r_sa + gamma * pV_sa

  assert qpi.shape == (env.nS, env.nA)
  return qpi

def softmax_policy_gradient(f_sa, s_n, a_n, adv_n):
  """
  Compute policy gradient of policy for discrete MDP, where probabilities
  are obtained by exponentiating f_sa and normalizing.
  
  See softmax_prob and softmax_policy_checkfunc functions in util. This function
  should compute the gradient of softmax_policy_checkfunc.
  
  INPUT:
    f_sa : a matrix representing the policy parameters, whose first dimension s 
           indexes over states, and whose second dimension a indexes over actions
    s_n : states (vector of int)
    a_n : actions (vector of int)
    adv_n : discounted long-term returns (vector of float)
  """
  row, col = f_sa.shape
  pi_axt = np.exp(f_sa) / np.sum(np.exp(f_sa), axis=1, keepdims=True)
  timeline = len(s_n)

  grad_sa = np.zeros((row, col))
  for i in range(timeline):
      grad_cur = np.zeros((row ,col))
      cur_s, cur_a = s_n[i], a_n[i]

      for a in range(4):
          if a == cur_a:
              t = pi_axt[cur_s, cur_a] - pi_axt[cur_s, cur_a] * pi_axt[cur_s, cur_a]
          else:
              t = -pi_axt[cur_s, a] * pi_axt[cur_s, cur_a]

          grad_cur[cur_s, a] = t

      grad_cur /= pi_axt[cur_s, cur_a]
      grad_sa += adv_n[i] * grad_cur

  assert grad_sa.shape == (row, col)
  grad_sa /= float(timeline)
  return grad_sa


def discount_reward(reward, gamma):
    T = len(reward)
    G = []
    for t in range(T):
        cur_sum = 0
        for k in range(t, T):
            cur_sum += np.power(gamma, k - t) * reward[k]

        cur_sum *= np.power(gamma, t)
        G.append(cur_sum)
    return G


def policy_gradient_step(policy, paths, gamma, stepsize):
  """
  Compute the discounted returns, compute the policy gradient (using softmax_policy_gradient above),
  and update the policy parameters policy.f_sa
  """
  theta = policy.f_sa
  row, col = theta.shape
  grad = np.zeros((row, col))
  for cur_path in paths:
      s, a = cur_path['observations'], cur_path['actions']

      dis_rew = discount_reward(cur_path['rewards'], gamma)  #scipy.signal.lfilter([1], [1, -gamma], s[::-1], axis=0)[::-1]
      grad += softmax_policy_gradient(theta, s, a, dis_rew)

  n = len(paths)
  grad /= float(n)
  policy.f_sa += stepsize * grad



