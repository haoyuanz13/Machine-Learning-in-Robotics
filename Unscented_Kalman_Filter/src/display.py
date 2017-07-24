import numpy as np
import matplotlib.pyplot as plt
import quaternions as quat
import gen_rot as rt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pylab as pl
import cv2, os

def plot(orientation):
	total = len(orientation)
	f = np.array(orientation)

	roll_f = f[:, 0]
	pitch_f = f[:, 1]
	yaw_f = f[:, 2]

	t = np.arange(0.0, total, 1.0)
	pl.figure(figsize = (16, 10), dpi = 60, facecolor = "white")


	pl.subplot(311)
	pl.title("UKF Estimated Orientation")
	pl.plot(t, roll_f, color = "red", label = "Roll")
	pl.legend(loc = "upper right")
	pl.xlabel("Normalized Time T")
	pl.ylabel("Angle /rad")
	pl.grid(True)

	pl.subplot(312)
	pl.plot(t, pitch_f, color = "blue", label = "Pitch")
	pl.legend(loc = "upper right")
	pl.xlabel("Normalized Time T")
	pl.ylabel("Angle /rad")
	pl.grid(True)

	pl.subplot(313)
	pl.plot(t, yaw_f, color = "green", label = "Yaw")
	pl.legend(loc = "upper right")
	pl.xlabel("Normalized Time T")
	pl.ylabel("Angle /rad")	
	pl.grid(True)
	pl.show()
