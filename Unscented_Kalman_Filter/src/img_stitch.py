import scipy;
from scipy import io;
import numpy as np;
import pdb;
import matplotlib.pyplot as plt;
import pylab as pl;
import cv2;

'''
	sphere to cat
'''
def sphere2Cat(azimuth, inclination, radius):
	theta = inclination;
	phi = azimuth;
	r = radius;
	x = r * np.cos(theta) * np.cos(phi);
	y = r * np.cos(theta) * np.sin(phi);
	z = r * np.sin(theta);
	num = theta.shape[0];
	pos = np.zeros((3, num));
	pos[0, :] = x.reshape(1, num);
	pos[1, :] = y.reshape(1, num);
	pos[2, :] = z.reshape(1, num);
	return x, y, z, pos;


'''
	cat to sphere transformation
'''
def cat2Sphere(world_pos):
	x = world_pos[0, :];
	y = world_pos[1, :];
	z = world_pos[2, :];
	r = np.sqrt(np.square(x) + np.square(y), np.square(z));
	theta = np.arctan2(z, r);
	phi = np.arctan2(y, x);
	azimuth = phi;
	inclination = theta;
	radius = r;
	return azimuth, inclination, radius;

'''
	main image stitching part
'''
def main():
	ukf_mode = 0;
	# load data
	im = scipy.io.loadmat('./cam/cam8.mat');
	imu = scipy.io.loadmat('./imu/imuRaw8.mat');

	# use ukf prediction
	if ukf_mode == 0:
		vicon = scipy.io.loadmat('./ukf_pred_rot/ukf_prediction8.mat');
		rot = vicon['ukf'];

	## use vicon
	else:
		vicon = scipy.io.loadmat('./vicon/viconRot2.mat');
		rot = vicon['rots'];

	cam = im['cam'];

	# normalize and match the timeline
	t_im = im['ts'];
	t_rot = imu['ts']; # the time in corresponding vicon
	im_num = t_im.shape[1];
	vicon_num = t_rot.shape[1];
	rot_selected = [];

	vicon_time_idx = 0;
	for i in range(im_num):
		camera_time = t_im[0, i];
		vicon_time = t_rot[0, vicon_time_idx];
		
		while(vicon_time < camera_time):
			vicon_time_idx += 1;
			
			if(vicon_time_idx >= vicon_num):
				vicon_time_idx = vicon_time_idx - 1;
				vicon_time = t_rot[0, vicon_time_idx];
				break;
			
			vicon_time = t_rot[0, vicon_time_idx];
		
		# after the while loop, we met the first vicon time > camera time
		
		rot_selected.append(vicon_time_idx);

	# rotate image according rotation matrix;
	row = 240;
	col = 320;
	pix = row * col;
	x = np.linspace(0, col - 1, col, dtype = int); #col
	y = np.linspace(0, row - 1, row, dtype = int); #row

	xx, yy = np.meshgrid(x, y);
	xx = xx.transpose().reshape(pix, 1);
	yy = yy.transpose().reshape(pix, 1);

	azimuth = (159 - xx) * (np.pi / 3 / col);
	inclination = (119 - yy) * (np.pi / 4 / row);

	x_cat, y_cat, z_cat, pos = sphere2Cat(azimuth, inclination, 1);
	# pos = np.array([x_cat, y_cat, z_cat]);

	# stitch image
	time_step = im_num;
	im_sti = np.zeros((960, 1920, 3));

	# stitch each frame in the current image
	for i in range(time_step):
		print i;
		im_cur = cam[:, :, :, i];
		rot_idx = rot_selected[i];
		
		if ukf_mode == 0:
			rot_cur = rot[rot_idx, :, :];
			rot_num = rot.shape[0];
		else:
			rot_cur = rot[:, :, rot_idx];
			rot_num = rot.shape[2];
		if rot_idx >= rot_num:
			break;		

		world_pos = np.dot(rot_cur, pos);
		azi, inc, r = cat2Sphere(world_pos)

		theta = np.floor((azi + np.pi) / (np.pi / 3 / col));
		height = np.floor(960 - (inc + (np.pi / 2)) / (np.pi / 4 / row));

		# traverse each pixel
		for j in range(pix):
			new_x = int(theta[j]);
			new_y = int(height[j]);
			if new_x < 0 or new_y < 0:
				continue;
			ori_x = xx[j, :];
			ori_y = yy[j, :];

			im_sti[new_y, new_x, :] = im_cur[ori_y, ori_x, :]
			im_sti[new_y, new_x, :].reshape(-1, 3);
		
		# print im_sti[new_y, new_x, :], im_cur[ori_y, ori_x, :];
		# im_sti = np.array(im_sti, dtype=np.uint8)
		# im_sti = cv2.cvtColor(im_sti, cv2.COLOR_BGR2RGB)
		# cv2.imwrite('color_img1.jpg', im_sti)


	im_sti = np.array(im_sti, dtype=np.uint8)
	im_sti = cv2.cvtColor(im_sti, cv2.COLOR_BGR2RGB)
	plt.imshow(im_sti);
	cv2.imwrite('panaroma_stirch.jpg', im_sti)
	plt.show();



if __name__ == '__main__':
	main();
