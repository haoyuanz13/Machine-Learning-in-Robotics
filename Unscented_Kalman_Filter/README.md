There are total six code files in the zip.

1. If you want to implement testing, run the 'UKF.py' directly, please make sure all test data store in a folder with the same directory as my code files.

2. For each test IMU data, you will see a figure display on the screen to show my filter result, once you close the figure, it will filter next test data in the folder automatically. 

3. The average filtering time may take 10 ~ 15 sec.

4. If you want to check my result only using prediction step, please increase the scale factor of noiseR (measurement noise) to a large number such as 1000000. (sclar factor locates at the line 49 in my 'UKF.py' code). 

5. With large value scale factor, it will increase time to perform filtering each round, might cost 20 ~ 30 sec.

6. The stitch image code 'img_stitch.py' in Python version will cost about 30 mins to generate one panorama, if you want to check the process, please run Matlab version 'img_stitch.m', they are both written via the same algorithm.

7. When executing 'img_stitch.m', just change the name of cam data and imu data as you wish.

Thanks for your time~
