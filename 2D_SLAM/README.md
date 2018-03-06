# Fast SLAM based on Particle Filter

The package creates Fast SLAM algorithm based on the particle filter. Finally, it can implement the structure of mapping and           localization in an indoor environment using information from an IMU and range sensors.       

The main algorithm refers to this tutorial slides: [A tutorial on fast slam](http://ais.informatik.uni-freiburg.de/teaching/ws12/mapping/pdf/slam10-fastslam-4.pdf).

Data
-----
All data are collected using IMU sensor reading from gyroscopes and accelerometers that describe the arm motions associated with the movements. The datasets were collected from a consumer mobile device so there is no need to consider bias or sensitivity issues. The data format as(**7d vector**): [ts, Ax, Ay, Az, Wx, Wy, Wz].      

Below figure shows the six different motions(1.Wave; 2.Infinity; 3.Eight; 4.Circle; 5.Beat3; 6.Beat4)
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
Add the package to your catkin workspace, execute the 'SLAM.py' directly, you are supposed tp see a figure on the screen which can show the mapping process. 

Due to the size of training data, it might be slow to show the moving. You can add some interval in order to see the significant mapping process.

Locate at line 112, 'SLAM.py', follow below operation:     
change
~~~~
for i in range(1, timeline): 
~~~~
to
~~~~
for i in range(1, timeline, <interval, e.g.25>)
~~~~

Result
------

Take a look at result folder and report file to see the map result and corresponding algorithm analysis.
