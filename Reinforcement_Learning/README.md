# Reinforcement Learning using Policy Gradient Methods          
The package simulates a game case called _Frozen Lake Problem_:           

Winter is here. You and your friends were tossing around a frisbee at the park when you made a wild throw that left the frisbee out in the middle of the lake. The water is mostly frozen, but there are a few holes where the ice has melted. If you step into one of those holes, you'll fall into the freezing water. At this time, there is an international frisbee shortage, so it is absolutely imperative that you navigate across the lake and retrieve the disc. However, the ice is slippery, so you won’t always move in the direction you intend...      
This package will implement several methods for dealing with the Frozen Lake problem. The situation can be represented via a _Markov Decision Process(MDP)_ and a strategy for retrieving the frisbee can be obtained using **value iteration (VI)**, **policy iteration (PI)**, and **policy gradient optimization (PGO)**.      

Moreover, once completed learning to retrieve the frisbee, feel free to use your PGO implementation as well as a more sophisticated proximal policy optimization to learn how to balance a pendulum or to play Atari games. 


Execution
---------
Add the package to your catkin workspace as well as its dependenices. Then execute **_'run_FrozenLake.py'_** directly. In the main file, the default grid space is _4 by 4_, you can change to _8 by 8_ alternatively.         

In order to speed up the convergence process (8 by 8 grid case), please change below arguments in the _'run_FrozenLake.py'_ (from line 114 to 122):        
1. **step size:** 1000.
2. **iteration time:** 500.
3. **horizaon:** 30.



Policy Gradient Optimal 
-----------------------
In order to solve the _Cart-Pole_ problem, which is also a typical reinforcement learning problem. Feel free to run **_run_pgo_continuous.py_** directly. You are supposed to see the Cart-Pole figure with iterative results.


Results
-------
All generated policies and training iterations are stored in the folder: **_results_**, and feel free to check the optimization results and corresponding analysis are in the **_Report.pdf_**.        

The below series of figures show the policy converging process in the _4-by-4_ grid cell.       

<p >
  <img src = "./results/vi1.png?raw=true" width="120" height="120">
  <img src = "./results/vi2.png?raw=true" width="120" height="120">
  <img src = "./results/vi3.png?raw=true" width="120" height="120">
  <img src = "./results/vi4.png?raw=true" width="120" height="120">
  <img src = "./results/vi5.png?raw=true" width="120" height="120">
  <img src = "./results/vi6.png?raw=true" width="120" height="120">
  <img src = "./results/vi7.png?raw=true" width="120" height="120">
</p>

