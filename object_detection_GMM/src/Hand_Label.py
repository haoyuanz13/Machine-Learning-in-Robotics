import numpy as np
import cv2, os
import pylab as pl
from roipoly import roipoly
from matplotlib import pyplot as plt


# store pixel within in barrel region
def getdata(mask, imgx, row, col):
  for i in range (row):
    for j in range (col):
      if mask[i, j]:
        res.append([imgx[i, j, 0], imgx[i, j, 1], imgx[i, j, 2]])


def main():
  # main code
  folder = "otherRed2"  # the folder stores training images
  res = []
  for filename in os.listdir(folder):
    # the default read-in channel order is BGR
    img = cv2.imread(os.path.join(folder, filename))
    imgx = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # use HSV color base
    
    row = imgx.shape[0]
    col = imgx.shape[1]

    # plot out the image
    pl.imshow(imgx, interpolation='nearest', cmap="hsv")
    pl.colorbar()
    pl.title("left click: line segment         right click: close region")

    # draw ROI
    ROIim = roipoly(roicolor = 'r')
    getdata(ROIim.getMask(imgx[:, :, 0]), imgx, row, col)


  # store training data
  np.save('data_redOtherK2.npy', res)


if __name__ == "__main__":
  main()
