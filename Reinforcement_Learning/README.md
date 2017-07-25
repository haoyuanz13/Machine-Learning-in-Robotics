The algorithm simulates a waliking robot on a frozen lake, every frozen place is safe and water interface is dangerous which will harm the robot. The package used Markov Decision Process to get optimal trajectory with highest reward, and policy gradient method to estimate optimal decision in the Cart-pole game.

Markov Decision Process
-----------------------
Execute 'run_FrozenLake.py' directly. The default grid is 4*4, in order to speed up the convergence process, if you want to run the frozen lake with 8*8 size, please change below argumets in the 'run_FrozenLake.py', from line 114 to 122, set step size as 1000, iteration time as 500, and horizaon as 30. 

Policy Gradient Optimal 
-----------------------
For choose Cart - pole problem. Please run 'run_pgo_continuous.py' directly. You are supposed to see the cart - pole figure as well as iteration results.

