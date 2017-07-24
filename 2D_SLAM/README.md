There are total four code files in the zip, 'SLAM.py', 'TextureMap.py', 'load_data.py' and 'MapUtils.py'.

1. Please run the 'SLAM' directly, you will see a figure on the screen which can show the mapping process, of course, it might be slow to show the moving since I don't skip any time stamp. You can add some interval at line 228 in order to see the significant mapping process.
e.g. change 'for i in range(1, total):' to 'for i in range(1, total, 25):'.

2. The 'SLAM' will cost almost 30 mins to complete data0, 2, and 3, and almost 80 mins to finish data1 and test data. If you want to run my code to ckeck the result, please comment the plot lines, from line 205 to 211 and line 263 to 266.

3. I have tried my best to write the TextureMap code but it might be wrong in some case, please feel free to have a look. If you want to test its accuracy, just input a time stamp into the main function. The problem might be the transferring coordinates from IR camera to RGB camera, I will try to fix that.

4. You can find all map image in the 'result' file.

Thanks for your time~