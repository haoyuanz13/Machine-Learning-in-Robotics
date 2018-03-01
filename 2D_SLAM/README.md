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
