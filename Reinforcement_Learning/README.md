Introduction
------------
The algorithm simulates a waliking robot on a frozen lake, only the frozen place is safe while other region is dangerous that will harm the robot. The main aim is to compute an optimal trajectory for robot from the start point to the target terminal with the highest reward. 

Usage
-----
Add the package to your catkin workspace as well as its dependenices. 

Markov Decision Process
-----------------------
Execute 'run_FrozenLake.py' directly. The default grid is 4 by 4, you can change to 8 by 8 size. In order to speed up the convergence process, please change below argumets in the 'run_FrozenLake.py':

From line 114 to 122, set step size as 1000, iteration time as 500, and horizaon as 30. 

Policy Gradient Optimal 
-----------------------
For solving Cart - pole issue, that's also a typical reinforcement learning problem. 

Please run 'run_pgo_continuous.py' directly. You are supposed to see the cart - pole figure as well as iteration results.

Results
-------
Optimization results and corresponding analysis are in the report, just feel free to take a look at that if necessary.
