# ML_in_Robotics
Machine learning algorithms applied into real modern robot.

The packaged includes several most popular used machine learning algorithm applied in the real robot, all algorithm 
and models are trained based on the data obtained from real robot. Those algorithms are the fundamental structures for the visual SLAM project. 

All packages include reports so that for better algorithm clarification and result showing.

# 1. Object Detection
Detect interested objects in the image via GMM and GM model, also measure the physical information of the objects in the real world.

# 2. UKF
Completed Unscented Kalman Filter to estimate the robot orientation, and finally built a panorama with images.

# 3. 2D SLAM
Based on the lidar scan data from the ground robot, using log-odds grid map to estimate 2D map and simulate robot walking trajectory.

# 4. Reinforcement Learning
Implemented policy gradient method to estiamte the optimal trajectory of robot within a certain environment with step rewards, also built 
deep network using tensorflow to train and solve Cart-pole game problem.
