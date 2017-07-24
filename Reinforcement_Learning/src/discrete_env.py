import numpy as np, numpy.random as nr
from scipy import special


from gym import Env, spaces
from gym.utils import seeding

def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()

def cat_sample(prob_nk):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    assert np.allclose(prob_nk.sum(axis=1,keepdims=True),1)
    N = prob_nk.shape[0]
    csprob_nk = np.cumsum(prob_nk, axis=1)
    out = np.zeros(N,dtype='i')
    for (n, csprob_k, r) in zip(range(N), csprob_nk, nr.rand(N)):
        for (k,csprob) in enumerate(csprob_k):
            if csprob > r:
                out[n] = k
                break
    return out

def cat_entropy(p):
    """
    Entropy of categorical distribution
    """
    # the following version has problems for p near 0
    #   return (-p * np.log(p)).sum(axis=1)
    return special.entr(p).sum(axis=1) #pylint: disable=E1101

def cat_kl(p, q):
    # the following version has problems for p near 0
    #   return (p*np.log(p/q)).sum(axis=1)
    return special.kl_div(p,q).sum(axis=1) #pylint: disable=E1101

class DiscreteEnv(Env):
    """
    Has the following members
    - nS: number of states
    - nA: number of actions
    - P: transitions (*)
    - isd: initial state distribution (**)

    (*) dictionary dict of dicts of lists, where
      P[s][a] == [(probability, nextstate, reward, done), ...]
    (**) list or array of length nS
    """
    def __init__(self, nS, nA, P, isd):
        self.P = P
        self.isd = isd
        self.lastaction=None # for rendering
        self.nS = nS
        self.nA = nA

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction=None
        return self.s

    def _step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d= transitions[i]
        self.s = s
        self.lastaction=a
        return (s, r, d, {"prob" : p})
