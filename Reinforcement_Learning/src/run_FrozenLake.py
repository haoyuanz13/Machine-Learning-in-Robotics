import numpy as np
import frozen_lake as FL
from gym.spaces import prng
import numdifftools as ndt
import todo as todo
import matplotlib.pyplot as plt
#import pdb

# Load Environment
def load_frozen_lake():
  env = FL.FrozenLakeEnv()
  env.seed(0)
  prng.seed(10)
  np.random.seed(0)
  np.set_printoptions(precision=3)
  print(env.__doc__)
  env.demonstrate()
  return env

def plot_V_pi(Vs, pis, env):
  for (V, pi) in zip(Vs, pis):
    plt.figure(figsize=(3,3))
    plt.imshow(V.reshape(4,4), cmap='gray', interpolation='none', clim=(0,1))
    ax = plt.gca()
    ax.set_xticks(np.arange(4)-.5)
    ax.set_yticks(np.arange(4)-.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    Y, X = np.mgrid[0:4, 0:4]
    a2uv = {0: (-1, 0), 1:(0, -1), 2:(1,0), 3:(-1, 0)}
    Pi = pi.reshape(4,4)
    for y in range(4):
      for x in range(4):
        a = Pi[y, x]
        u, v = a2uv[a]
        plt.arrow(x, y,u*.3, -v*.3, color='m', head_width=0.1, head_length=0.1) 
        plt.text(x, y, str(env.desc[y,x].item().decode()),
                 color='g', size=12,  verticalalignment='center',
                 horizontalalignment='center', fontweight='bold')
    plt.grid(color='b', lw=2, ls='-')

def show_V_pi(Vs, pis):
  nIt = len(Vs)
  print("Iteration | max|V-Vprev| | # chg actions | V[0]")
  print("----------+--------------+---------------+---------")
  for it in range(1,nIt):
    max_diff = np.abs(Vs[it] - Vs[it-1]).max()
    nChgActions="N/A" if it is 1 else (pis[it-1] != pis[it-2]).sum()
    print("%4i      | %6.5f      | %4s          | %5.3f"%(it, max_diff, nChgActions, Vs[it][0]))
  print("\n")

def test_VI(env, gamma, nIt):
  Vs_VI, pis_VI = todo.value_iteration(env, gamma, nIt)  # TODO
  # sm.showGrid(env, Vs_VI, pis_VI)
  Vpi = todo.policy_evaluation_v(pis_VI[15], env, gamma) # TODO
  # display
  print("From value iteration", Vs_VI[15])
  print("From policy evaluation", Vpi)
  print("Difference", Vpi - Vs_VI[15])
  print("\n")
  show_V_pi(Vs_VI, pis_VI)
  plot_V_pi(Vs_VI[:10], pis_VI[:10], env)
  plt.figure()
  plt.plot(Vs_VI)
  plt.title("Values of different states")  


def test_PI(env, gamma, nIt):
  Qpi = todo.policy_evaluation_q(np.arange(env.nS), env, gamma) # TODO
  Vs_PI, pis_PI = todo.policy_iteration(env, gamma, nIt)        # TODO
  # display
  print("Qpi:\n", Qpi)
  print("\n")
  show_V_pi(Vs_PI, pis_PI) 
  plt.figure()
  plt.plot(Vs_PI)
  plt.title("Values of different states")
  Vs_VI, pis_VI = todo.value_iteration(env, gamma, nIt)
  for s in range(5):
    plt.figure()
    plt.plot(np.array(Vs_VI)[:,s])
    plt.plot(np.array(Vs_PI)[:,s])
    plt.ylabel("value of state %i"%s)
    plt.xlabel("iteration")
    plt.legend(["value iteration", "policy iteration"], loc='best')  

def test_policy_gradient(env):
  from util import softmax_policy_checkfunc
  policy = FL.RandomDiscreteActionChooser(env.nA)
  rdata = FL.rollout(env, policy, 100)
  print (rdata)
  s_n = rdata['observations'] # Vector of states (same as observations since MDP is fully-observed)
  a_n = rdata['actions'] # Vector of actions (each is an int in {0,1,2,3})
  n = a_n.shape[0] # Length of trajectory
  q_n = np.random.randn(n) # Returns (random for the sake of gradient checking)
  f_sa = np.random.randn(env.nS, env.nA) # Policy parameter vector. explained shortly.  
  
  # Compute numerical gradients and compare to the analytical computation from
  # todo.softmax_policy_gradient in order to verify your implementation
  stepdir = np.random.randn(*f_sa.shape)
  auxfunc = lambda x: softmax_policy_checkfunc(f_sa+stepdir*x, s_n, a_n, q_n)
  numgrad = ndt.Derivative(auxfunc)(0)
  g = todo.softmax_policy_gradient(f_sa, s_n, a_n, q_n) # TODO
  anagrad = (stepdir*g).sum()

  assert abs(numgrad - anagrad) < 1e-10
  print("The numgrad value is: ", numgrad)
  print("The anagrad value is: ", anagrad)


def test_PGO(env, gamma):
  policy = FL.FrozenLakeTabularPolicy(env.nS)
  stat2ts = todo.policy_gradient_optimize(env, policy,
                gamma=gamma,
                max_pathlength=100,
                timesteps_per_batch=2000,
                n_iter=100,
                stepsize=100)  #
  # change iteration times, step size and horizon value when apply 8 * 8 grid
  # Display
  FL.animate_rollout(env, policy, delay=.001, horizon=10)
  plt.figure()
  plt.title("Episode Reward")
  EpRewMean = np.array(stat2ts["EpRewMean"])
  EpRewStd = np.array(stat2ts["EpRewSEM"])
  plt.errorbar(np.arange(len(EpRewMean)), EpRewMean, yerr=EpRewStd, errorevery=5, linewidth=1)
  plt.figure()
  plt.title("Mean Episode Length")
  plt.plot(stat2ts["EpLenMean"])
  plt.figure()
  plt.title("Perplexity")
  plt.plot(stat2ts["Perplexity"])
  plt.figure()
  plt.title("Mean KL Divergence Between Old & New Policies")
  plt.plot(stat2ts["KLOldNew"])
  plt.show()


def test_frozen_lake():
  env = load_frozen_lake()
  GAMMA = 0.95 # we'll be using this same value in subsequent problems
  nIt = 20     # number of iterations for Value Iteration
  
  # TODO:
  test_VI(env, GAMMA, nIt)
  test_PI(env, GAMMA, nIt)

  test_policy_gradient(env)
  test_PGO(env, GAMMA)
  #plt.show()

if __name__ == "__main__":
  test_frozen_lake()
  



