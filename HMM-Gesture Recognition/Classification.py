import numpy as np
from hmm import HMM
import os
import readinData as rd

folder = 'models'
name_model = []
T_set, B_set, Pi_set = [], [], []

# read in models
for filename in os.listdir(folder):
  name_model.append(filename)
  model_cur = np.load(os.path.join(folder, filename))
  param = model_cur.all()

  T_set.append(param['T'])
  B_set.append(param['B'])
  Pi_set.append(param['Pi'])

model_total = len(T_set)

# test
folder = 'single'  # folder stores test data sets
obs_set, num_data = {}, 0
N, M = 15, 40

for txt in os.listdir(folder):
  ts, obs = rd.disData(folder, txt, M)
  obs_set[num_data] = obs.astype(np.int16)
  num_data += 1

res = []
for ind_data in range(num_data):
  max_log = -10000000000000
  detect_class = ''
  obs = obs_set[ind_data]

  for i in range(model_total):
    hmm = HMM(T_set[i], B_set[i], Pi_set[i], N, M)
    log_cur = hmm.forward(obs)

    if log_cur > max_log:
      max_log = log_cur
      detect_class = name_model[i][:-4]

  res.append(detect_class)

print res



