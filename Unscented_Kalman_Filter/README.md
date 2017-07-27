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
