# Orientation Estimation using Unscented Kalman Filter(UKF)

This package implements a Kalman filter to track three dimensional orientation of a hand-handled camera. Given IMU sensor readings from gyroscopes and accelerometers, the algorithm will estimate the underlying 3D orientation by learning the appropriate model parameters from ground truth data given by a Vicon motion capture system. Then it's able to generate a real-time panoramic image from camera images using the 3D orientation filter.       

The main UKF algorithm refers to this paper: [A Quaternion-based Unscented Kalman Filter for Orientation Tracking](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1257247).



Data
-----
All data are collected using IMU sensor reading from gyroscopes and accelerometers that describe the arm motions associated with the movements. Those data are in raw version such that it's necessary to consider bias or sensitivity issues. The data format as(**6d vector**): [Ax, Ay, Az, Wz, Wx, Wy].      

Below figure shows more intuitive camera frame model:     
<div align=center>
  <img width="500" height="250" src="./gesture_fig.png", alt="gesture"/>
</div>

Total three datasets in the package:     
1. _train_: contains all training data.
2. _test_single_: contains test dataset using single step motion for each gesture.
3. _test_multiple_: contains test dataset using multiple steps motion for each gesture.


Execution
---------
1. _training.py_: the main training file to train the HMM model for each gesture type.
2. _hmm.py_: the general HMM model.
3. _Classification.py_: the test file.
4. _utils.py_: helper functions including save, loading data and k-means.



Results and Report
-------
1. _hmm_trained_models_: contains all trained models as well as k-means clustered results.
2. _hmm_test_res_: contains all test results figures to show the prediction type with corresponding confidence.

**_Training Results_**
<div align=center>
  <img width="500" height="500" src="./hmm_test_res/training.png", alt="training results"/>
</div>

**_Test (single) Results_** 
<div align=center>
   <img width="500" height="130" src="./hmm_test_res/test_single.png", alt="test(single) results"/>
</div>

**_Test (multiple) Results_** 
<div align=center>   
   <img width="500" height="130" src="./hmm_test_res/test_single.png", alt="test(single) results"/>
</div>


Please feel free to execute the file _Classification.py_ to see more intuitive test results.

In addition, you can check the report _GestureRecoHMM_report.pdf_ for more detailed explanantions.


Usage
-----
Add the package to your catkin workspace and its dependencies. Execute the 'UKF.py', please make sure all test data store in a folder with the same directory as the code files. 

For each test IMU data, you are supposed to see a figure display on the screen to show filter result in roll, pitch and yaw. Once you close the figure, it will filter next test data in the folder automatically. 

Image stitching to build panorama
---------------------------------
The image stitching code file 'img_stitch.py' in Python version will cost about 30 mins to generate one panorama, if you want to check the process, please run Matlab version 'img_stitch.m', they are both written via the same algorithm. 

Results
-------
The UKF filtered robot pose and stitching image results are all in the report, you can also refer to algorithm details in that as well. 
