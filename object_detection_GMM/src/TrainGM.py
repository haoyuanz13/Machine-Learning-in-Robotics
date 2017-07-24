import numpy as np

barr = np.load('data_redBarrel.npy') # read training data in
barrel = np.array(barr, dtype = int)
size = len(barrel[:])

# compute mean value        
mean = np.sum(barrel, axis = 0) / float(size)
# compute covariance matrix 
corvar = np.zeros((3, 3))
for i in range (size):
	norm = barrel[i, :] - mean
	corvar += np.outer(norm, norm)
corvar = corvar / float(size)

# alternative computation
a_m = barrel - mean
corvar1 = np.dot(a_m.transpose(), a_m) / float(size)

np.save('model_mean.npy', mean)
np.save('model_var.npy', corvar)
