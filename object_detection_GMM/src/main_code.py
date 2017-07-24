from __future__ import division
from skimage import measure, draw, data, filters, segmentation, measure, morphology, color
from skimage.measure import label
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pylab as pl
import cv2, os
import math
import copy

# compute probability of pixels in image
def sub_prob(m, v, w, x):
    det_v = np.linalg.det(v) # det value of variance
    inv_v = np.linalg.inv(v) # inverse value of variance

    # Gaussian model 
    term_a = (-1.5) * math.log(2 * math.pi)
    term_b = (-0.5) * math.log(det_v)

    norm = np.matrix(x - m)
    x = np.sum(np.multiply(norm * inv_v, norm), axis = 1) / float(-2)
    prob = term_a + term_b + x

    res = np.multiply(np.exp(prob), w)
    return res

# determine whether current pixel belongs to red barrel or not
def label_redB(ma, threshold):
    sum_prob_rb = np.zeros((row * col, 1))
    sum_prob_rc = np.zeros((row * col, 1))
    sum_prob_ro1 = np.zeros((row * col, 1))
    sum_prob_ro2 = np.zeros((row * col, 1))

    # prob of red barrel
    for i in range (cluster_rb):
        cur_mean = dic_rb[i]["mean"]
        cur_var = dic_rb[i]["covar"]
        cur_wei = dic_rb[i]["wei"]

        sum_prob_rb += sub_prob(cur_mean, cur_var, cur_wei, ma)    
    # prob of red chairs
    for i in range (cluster_rc):
        cur_mean = dic_rc[i]["mean"]
        cur_var = dic_rc[i]["covar"]
        cur_wei = dic_rc[i]["wei"]

        sum_prob_rc += sub_prob(cur_mean, cur_var, cur_wei, ma)
    # prob of other red1
    for i in range (cluster_ro1):
        cur_mean = dic_ro1[i]["mean"]
        cur_var = dic_ro1[i]["covar"]
        cur_wei = dic_ro1[i]["wei"]

        sum_prob_ro1 += sub_prob(cur_mean, cur_var, cur_wei, ma)
    # prob of other red2
    for i in range (cluster_ro2):
        cur_mean = dic_ro2[i]["mean"]
        cur_var = dic_ro2[i]["covar"]
        cur_wei = dic_ro2[i]["wei"]

        sum_prob_ro2 += sub_prob(cur_mean, cur_var, cur_wei, ma)
    # set different weight of each GMMs
    mask = (sum_prob_rb > threshold) & (sum_prob_rb > sum_prob_rc) & (sum_prob_rb >= 0.05 * sum_prob_ro1) & (sum_prob_rb >= 0.12 * sum_prob_ro2)
    res = mask.reshape(row, col)
    return res

# permutate all possible ROI combinations
def dfs_find(pos, num, pixel, arr, start, total_pixel, num_comb, pos_res):
    if num_comb == 0:
        ltr, ltc, rbr, rbc = arr
        height = rbr - ltr
        width = rbc - ltc
        
        if (height * width) > 2.5 * total_pixel:
            return

        ratio = max(height / width, width / height)
        if ratio <= 1 or ratio >= 2:
            return

        pos_res.append(copy.copy(arr))
        return
    else:
        for i in range(start, num - num_comb + 1):
            # merge contours 
            minR = pos[i][0] if len(arr) == 0 else min(arr[0], pos[i][0])
            minC = pos[i][1] if len(arr) == 0 else min(arr[1], pos[i][1])
            maxR = pos[i][2] if len(arr) == 0 else max(arr[2], pos[i][2])
            maxC = pos[i][3] if len(arr) == 0 else max(arr[3], pos[i][3])

            dfs_find(pos, num, pixel, [minR, minC, maxR, maxC], i + 1, total_pixel + pixel[i], num_comb - 1, pos_res)

# image processing
def pos_barrel(t):
    # set threshold to seperate regions
    thresh = filters.threshold_otsu(t)
    bw = morphology.closing(t > thresh, morphology.square(3))

    cleared = bw.copy()
    segmentation.clear_border(cleared)

    label_image =measure.label(cleared)
    borders = np.logical_xor(bw, cleared)
    label_image[borders] = -1
    image_label_overlay =color.label2rgb(label_image, image = t)

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize = (8, 8))

    ax0.imshow(img_ori, plt.cm.jet)
    ax1.imshow(cleared, plt.cm.gray)
    ax2.imshow(image_label_overlay)

    pos_res = [] # store all posiible contour candidates
    pos_set = [] # store all detected contours 
    num_pixel = [] # store number of pixels within contour area 
    for region in measure.regionprops(label_image):
        if region.area < 1000:
            continue
        pos_set.append(region.bbox)
        num_pixel.append(region.area)

    # apply DFS to find all barrel candidates 
    num_contour = len(pos_set)
    for i in range (num_contour):
        dfs_find(pos_set, num_contour, num_pixel, [], 0, 0, i + 1, pos_res)
    # no barrel in image 
    if len(pos_res) == 0:
        return []
    # handle overlap contours
    true_barrel = [True] * len(pos_res)
    for i in range (len(pos_res)):
        if not true_barrel[i]:
            continue;
        cur_minr, cur_minc = pos_res[i][0], pos_res[i][1]
        cur_maxr, cur_maxc = pos_res[i][2], pos_res[i][3]
        for j in range (len(pos_res)):
            if (i == j) or (not true_barrel[j]):
                continue;
            lt_side = (cur_minr >= pos_res[j][0]) and (cur_minc >= pos_res[j][1])
            rb_side = (cur_maxr <= pos_res[j][2]) and (cur_maxc <= pos_res[j][3])
            # find overlap contour
            if lt_side and rb_side:
                ratio_i = (cur_maxr - cur_minr) / (cur_maxc - cur_minc)
                ratio_j = (pos_res[j][2] - pos_res[j][0]) / (pos_res[j][3] - pos_res[j][1])

                diff_i = math.fabs(ratio_i - 1.5)
                diff_j = math.fabs(ratio_j - 1.5)

                if diff_i > diff_j:
                    true_barrel[i] = False
                    break
                else:
                    true_barrel[j] = False
    # draw contours for detected true barrel region
    set_barrel = []
    for i in range (len(pos_res)):
        if not true_barrel[i]:
            continue

        minr, minc, maxr, maxc = pos_res[i]
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                              fill = False, edgecolor = 'red', linewidth = 2)
        ax2.add_patch(rect)
        # count distance
        img_height = maxr - minr
        distance = 1.185 * 513 / float(img_height)
        set_barrel.append([minr, minc, maxr, maxc, distance])
    
    fig.tight_layout()
    plt.show()
    return set_barrel
    
# detect red barrel in image
def detect(imgx):    
    img_temp = np.zeros((row * col, 3))
    img_temp[:, 0] = imgx[:, :, 0].reshape(1, row * col)
    img_temp[:, 1] = imgx[:, :, 1].reshape(1, row * col)
    img_temp[:, 2] = imgx[:, :, 2].reshape(1, row * col)

    table = np.array(label_redB(img_temp, threshold), dtype = int)
    res = pos_barrel(table)
    return res

#************************* main code *************************#
dic_rb = np.load('GMM_redBarrel.npy')
dic_rc = np.load('GMM_redChair.npy')
dic_ro1 = np.load('GMM_redOther1.npy')
dic_ro2 = np.load('GMM_redOther2.npy')
cluster_rb = len(dic_rb)
cluster_rc = len(dic_rc)
cluster_ro1 = len(dic_ro1)
cluster_ro2 = len(dic_ro2)

folder = "folder_name"

threshold = 1e-10
for filename in os.listdir(folder):    
    img_ori = cv2.imread(os.path.join(folder, filename))    
    img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2HSV)

    row = img.shape[0]
    col = img.shape[1]

    pos_inf = detect(img)
    # print detected result
    print ("Detected total %d red barrel(s) in the image. \n" %len(pos_inf))
    for pos in pos_inf:
        print "Left Top position is:", [pos[0], pos[1]]
        print "Left Bottom position is:", [pos[2], pos[1]]
        print "Right Top position is:", [pos[0], pos[3]]
        print "Right Bottom position is:", [pos[2], pos[3]], "\n"
        print ("The barrel is %f meters away from the camera. \n" %pos[4])

    print "************* next image *************"




















