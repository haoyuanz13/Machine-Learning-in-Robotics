import sys
import numpy as np, numpy.random as nr
from six import StringIO, b
from util import softmax_prob
from gym import utils
import discrete_env

  
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ],
}

class FrozenLakeEnv(discrete_env.DiscreteEnv):
  """
  Winter is here. You and your friends were tossing around a frisbee at the park
  when you made a wild throw that left the frisbee out in the middle of the lake.
  The water is mostly frozen, but there are a few holes where the ice has melted.
  If you step into one of those holes, you'll fall into the freezing water.
  At this time, there's an international frisbee shortage, so it's absolutely imperative that
  you navigate across the lake and retrieve the disc.
  However, the ice is slippery, so you won't always move in the direction you intend.
  The surface is described using a grid like the following

      SFFF
      FHFH
      FFFH
      HFFG

  S : starting point, safe
  F : frozen surface, safe
  H : hole, fall to your doom
  G : goal, where the frisbee is located

  The episode ends when you reach the goal or fall in a hole.
  You receive a reward of 1 if you reach the goal, and zero otherwise.

  """

  metadata = {'render.modes': ['human', 'ansi']}

  def __init__(self, desc=None, map_name="4x4",is_slippery=True):
      if desc is None and map_name is None:
          raise ValueError('Must provide either desc or map_name')
      elif desc is None:
          desc = MAPS[map_name]
      self.desc = desc = np.asarray(desc,dtype='c')
      self.nrow, self.ncol = nrow, ncol = desc.shape

      nA = 4
      nS = nrow * ncol

      isd = np.array(desc == b'S').astype('float64').ravel()
      isd /= isd.sum()

      P = {s : {a : [] for a in range(nA)} for s in range(nS)}

      def to_s(row, col):
          return row*ncol + col
      def inc(row, col, a):
          if a==0: # left
              col = max(col-1,0)
          elif a==1: # down
              row = min(row+1,nrow-1)
          elif a==2: # right
              col = min(col+1,ncol-1)
          elif a==3: # up
              row = max(row-1,0)
          return (row, col)

      for row in range(nrow):
          for col in range(ncol):
              s = to_s(row, col)
              for a in range(4):
                  li = P[s][a]
                  letter = desc[row, col]
                  if letter in b'GH':
                      li.append((1.0, s, 0, True))
                  else:
                      if is_slippery:
                          for b in [(a-1)%4, a, (a+1)%4]:
                              newrow, newcol = inc(row, col, b)
                              newstate = to_s(newrow, newcol)
                              newletter = desc[newrow, newcol]
                              done = bytes(newletter) in b'GH'
                              rew = float(newletter == b'G')
                              li.append((0.8 if b==a else 0.1, newstate, rew, done))
                      else:
                          newrow, newcol = inc(row, col, a)
                          newstate = to_s(newrow, newcol)
                          newletter = desc[newrow, newcol]
                          done = bytes(newletter) in b'GH'
                          rew = float(newletter == b'G')
                          li.append((1.0, newstate, rew, done))

      super(FrozenLakeEnv, self).__init__(nS, nA, P, isd)

  def _render(self, mode='human', close=False):
      if close:
          return
      outfile = StringIO() if mode == 'ansi' else sys.stdout

      row, col = self.s // self.ncol, self.s % self.ncol
      desc = self.desc.tolist()
      desc = [[c.decode('utf-8') for c in line] for line in desc]
      desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
      if self.lastaction is not None:
          outfile.write("  ({})\n".format(["Left","Down","Right","Up"][self.lastaction]))
      else:
          outfile.write("\n")
      outfile.write("\n".join(''.join(line) for line in desc)+"\n")

      return outfile

  def demonstrate(self):
    print("    Let's look at a random episode...")
    self.reset()
    for t in range(100):
      self.render()
      a = self.action_space.sample()
      ob, rew, done, _ = self.step(a)
      if done:
        break
    assert done
    self.render()

    print("    In the episode above, the agent falls into a hole after two timesteps.")
    print("    Also note the stochasticity -- on the first step, the DOWN action is")
    print("    selected, but the agent moves to the right.")
    print("\n")
    
    print("Let us look at the transition model of the Frozen Lake Problem now.\n")
    print("env.P is a two-level dict where the first key is the state and the second key is the action.")
    print("The 2D grid cells are associated with indices [0, 1, 2, ..., 15] from left to right and top to down, as in")
    print(np.arange(16).reshape(4,4))
    print("env.P[state][action] is a list of tuples (probability, nextstate, reward).\n")
    print("For example, state 0 is the initial state, and the transition information for s=0, a=0 is \nP[0][0] =", self.P[0][0], "\n")
    print("As another example, state 5 corresponds to a hole in the ice, which transitions to itself with probability 1 and reward 0.")
    print("P[5][0] =", self.P[5][0], '\n')
    print("\n")


class Policy(object):
  def step(self, o):
    """
    Return dict including

    required: 
        a : actions
    optional:
        pa : specifies probability distribution that 'a' was sampled from
        [whatever else your learning algorithm will need]
    """
    raise NotImplementedError
   
class RandomDiscreteActionChooser(Policy):
  def __init__(self, n_actions):
      self.n_actions = n_actions
  def step(self, observation):
      return {"action":np.array([nr.randint(0, self.n_actions)])}
      
class FrozenLakeTabularPolicy(Policy):
  def __init__(self, n_states):
      self.n_states = n_states
      self.n_actions = n_actions = 4        
      self.f_sa = np.zeros((n_states, n_actions))

  def step(self, s_n):
      f_na = self.f_sa[s_n]
      prob_nk = softmax_prob(f_na)
      acts_n = discrete_env.cat_sample(prob_nk)
      return {"action": acts_n,
              "pdist" : f_na}

  def compute_pdists(self, s_n):
      return self.f_sa[s_n]

  def compute_entropy(self, f_na):
      prob_nk = softmax_prob(f_na)
      return discrete_env.cat_entropy(prob_nk)

  def compute_kl(self, f0_na, f1_na):
      p0_na = softmax_prob(f0_na)
      p1_na = softmax_prob(f1_na)
      return discrete_env.cat_kl(p0_na, p1_na)
      

def rollout(env, policy, max_pathlength):
    """
    Simulate the env and policy for max_pathlength steps
    """
    ob = env.reset()
    ob = np.array([ob])
    terminated = False

    obs = []
    actions = []
    rewards = []
    pdists = []
    for _ in range(max_pathlength):
        obs.append(ob)
        pol_out = policy.step(ob)
        action = pol_out["action"]        
        actions.append(action)
        pdists.append(pol_out.get("pdist",[None]))

        ob, rew, done, _ = env.step(action[0])
        ob = np.array([ob])
        rewards.append(rew)
        if done:
            terminated = True
            break
    return {"observations" : np.concatenate(obs), "pdists" : np.concatenate(pdists), 
        "terminated" : terminated, "rewards" : np.array(rewards), "actions" : np.concatenate(actions)}

def animate_rollout(env, policy, horizon=100, delay=0.05):
  """
  Do rollouts and plot at each timestep
  delay : time to sleep at each step
  """
  import time
  obs = env.reset()
  env.render()
  for i in range(horizon):
    a = policy.step(np.array([obs]))["action"]
    obs, _rew, done, _ = env.step(a[0])
    env.render()
    if done:
      print ("terminated after %s timesteps"%(i+1))
      break
    time.sleep(delay)
    

