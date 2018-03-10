# Object Detection (Red Barrel) using Gaussian Mixture Model

In this package, it trains a probabilistic color model from image data, which will be used to segment and detect a target of interest (red barrel), and find the relative world coordinates of the target with respect to the camera frame.          

More specifically, given a set of training images, hand-label examples of different colors first. Then from these training examples, build color classifiers for several colors (e.g., red, yellow, brown, etc.) and finally a red barrel detector. After detection, use own designed algorithm to obtain the bounding box of a detected barrel in the image frame and use the camera parameters to calculate the distance to the barrel on a set of new test images.        

The main GMM algorithm refers to this tutorial paper: [Mixture Model and EM](http://www.cse.psu.edu/~rtc12/CSE586Spring2010/papers/prmlMixturesEM.pdf).


Algorithm and Tips
------------------
1. **_Hand-label appropriate regions in the training images with discrete color labels_**                   
For this project, we will be especially interested in regions containing the red barrel (positive examples) and images containing similar colored-areas that are not a barrel (negative examples). If you are more ambitious, you could try to implement more automated ways of labeling images, e.g., by an unsupervised image segmentation, or an adaptive region flooding algorithm. Lighting invariance will be an issue, so you should think carefully about the best color space to use, and perhaps some low-level adaptation on the image.        
2. **_Use a learning algorithm to partition the color space into appropriate class color regions_**           
In this project, we use Gaussain Mixture Model(GMM) to repersent each possible color region, but you are also free to try other machine learning approaches if you are interested in, e.g., decision trees, support vector machines, etc. You need to make your algorithm so that it is able to robustly generalize to new images. To prevent overfitting the training images, split them into training and validation sets. Train your algorithms using the training set and evaluate their performance on the validation set. This will allow you to compare different parameters for the probabilistic models and different color space representations.          

3. **_Geometric Distance Estimation_**           
Once the color regions are identified, we use shape statistics and other higher-level features to decide where the barrel is located in the images. Use designed algorithms (e.g., camera model) to identify the coordinates of a bounding box for each detected barrel, which should compute an estimate of the distance to the barrel.        


Data
-----
All training data are images that contain at least one red barrel, some of them may only include red barrel without any other confusing red color, and some may contain more than one similar red color. Examples are shown below:

<p >
<align="left">
  <img src = "./data/red_barrel/2.14.png?raw=true" width="280" height="210">
<align="center">
  <img src = "./data/other_red1/2.8.png?raw=true" width="280" height="210">
<align="right">
  <img src = "./data/red_chair/2.6.png?raw=true" width="280" height="210">
</p>

Typically, for each train image, we manually crop and label red barrel data, as well as other interest color, and then store them as _.npy_ format. Total four dataset types in the package:     
1. _red_barrel_: images used for training red color model from red barrel.
2. _red_chair_: images used for training red color model from red chair.
3. _other_red1/2_: images used for training red color model from other red objects.       


Execution
---------
1. _UKF.py_: the main training file to estimate the camera pose.
2. _ADC.py_: works as dataloader and clean up raw data.
3. _quaternions.py_: contains all basic operations of quaternions (including the average value estimation algorithm).
4. _display.py_: plot helper function.
5. _img_stitch.py_: generate panorama given camera pose information and images. (**Optional**: I also provide the matlab image stitching version, _'img_stitch.m'_, which is much faster).         


For each test IMU data, you are supposed to see a figure display on the screen to show the filtered result in roll, pitch and yaw three dimension. Once you close the figure, it will work on the next coming test data in the target folder automatically. 


Results and Report
-------
All results are stored in the folder **_result_**, including:

**_3D Orientation Estimation_**
<div align=center>
  <img width="560" height="420" src="./result/ori_est.jpg", alt="rpy"/>
</div>

**_Generated Panorama_** 
<div align=center>
   <img width="650" height="380" src="./result/panorama.jpg", alt="panorama"/>
</div>

**_Real-time Panorama Generating Video_**         
The video is [here](https://drive.google.com/open?id=0B-YfsvV6PlJRaEtVb0pjTnNSaE0).



In addition, you can check the report **_Report.pdf_** for more detailed explanantions.

Usage
-----
Execute the 'main_code' directly to those test images. It's better to arrange all test images in a folder and then modify the filename in 'main_code'.

For each test image, you are supposed to see a figure shown in the screen which contains total three images, the first one is the original image (color converted), the second one contains all detected region candidates, and the third one is the contour of red barrel. When you close the figure, the number of red barrel, four corner positions and distance information will print out. 

Code files clarification 
----------
'main_code' is for testing images and find possible red barrel, it contains main part of this project, such as decision operator for each pixel and image processing.
'TrainGMM' is the code of training GMM via EM, and I use GMM to construct all models. 
'TrainGM' is for training GM.
'Hand_label' is for labeling ROI manually and store data in certain file.

Trained models
--------------
I trained total four GMMs in this project.
'GMM_redBarrel': model of red barrel.
'GMM_redChair': model of red chairs.
'GMM_redOther1': model of part of other red objects, including red robotics, red wall.
'GMM_redOther2': model of part of other red objects, including red ball and red bicycle.



