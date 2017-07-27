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



