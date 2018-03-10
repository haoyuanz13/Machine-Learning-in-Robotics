# Reinforcement Learning using Policy Gradient Methods          
The package simulates a game case called _Frozen Lake Problem_:           

Winter is here. You and your friends were tossing around a frisbee at the park when you made a wild throw that left the frisbee out in the middle of the lake. The water is mostly frozen, but there are a few holes where the ice has melted. If you step into one of those holes, you'll fall into the freezing water. At this time, there is an international frisbee shortage, so it is absolutely imperative that you navigate across the lake and retrieve the disc. However, the ice is slippery, so you won’t always move in the direction you intend...      
This package will implement several methods for dealing with the Frozen Lake problem. The situation can be represented via a _Markov Decision Process(MDP)_ and a strategy for retrieving the frisbee can be obtained using **value iteration (VI)**, **policy iteration (PI)**, and **policy gradient optimization (PGO)**.      

Moreover, once completed learning to retrieve the frisbee, feel free to use your PGO implementation as well as a more sophisticated proximal policy optimization to learn how to balance a pendulum or to play Atari games. 


Execution
---------
Add the package to your catkin workspace as well as its dependenices.       

Then run **_'run_FrozenLake.py'_** directly. In the main file, the default grid space is _4 by 4_, you can change to _8 by 8_ alternatively.         

In order to speed up the convergence process (8 by 8 grid case), please change below arguments in the _'run_FrozenLake.py'_:

**From line 114 to 122, set step size as 1000, iteration time as 500, and horizaon as 30**

Policy Gradient Optimal 
-----------------------
For solving Cart - pole issue, that's also a typical reinforcement learning problem. 

Please run 'run_pgo_continuous.py' directly. You are supposed to see the cart - pole figure as well as iteration results.

Results
-------
Optimization results and corresponding analysis are in the report, just feel free to take a look at that if necessary.
