# Gesture Recoginition using Hidden Markov Model (HMM)

The package creates an algorithm based on the HMM to recognize different robot arm motion gestures. Finally, it can achieve the ability to classify unknown arm motions in real-time.

Data
-----
All data are collected using IMU sensor reading from gyroscopes and accelerometers that describe the arm motions associated with the movements. The datasets were collected from a consumer mobile device so there is no need to consider bias or sensitivity issues. The data format as(**7d vector**): [ts, Ax, Ay, Az, Wx, Wy, Wz].

Execution
---------
1. _training.py_: the main training file to train the HMM model for each gesture type.
2. _hmm.py_: the general HMM model.
3. _Classification.py_: the test file.
4. _utils.py_: helper functions including save, loading data and k-means.
