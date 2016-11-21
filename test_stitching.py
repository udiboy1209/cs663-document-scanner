from docscanner.stitching import get_connectivity_mat, merge_simple
from docscanner.feature_matching import get_orb_features

import sys, os
import cv2
import numpy as np
from matplotlib import pyplot as plt

img_folder = sys.argv[1]
img_files = sorted([os.path.join(img_folder,f) \
                    for f in os.listdir(img_folder) \
                    if os.path.isfile(os.path.join(img_folder,f))])
img_files = img_files[0:2]
print(img_files)

imgs = [cv2.resize(cv2.imread(f,0), (0,0), fx=0.2, fy=0.2) for f in img_files]
imgs_small = []
for img in imgs:
    img_small = img.copy() # cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    avg = np.sum(img_small) / (img_small.shape[0] * img_small.shape[1])
    img_small[img_small>avg] = 255
    img_small[img_small<avg] = 0

    imgs_small.append(img_small)

features = [get_orb_features(img, 20000) for img in imgs_small]

# for im,f in zip(imgs,features):
#     img_kp = np.empty(0)
#     img_kp = cv2.drawKeypoints(im,f[0],outImage=img_kp,color=(0,255,0),flags=0)
#     plt.figure(),plt.imshow(img_kp)

# plt.show()

matches_list = get_connectivity_mat(imgs_small, features)
img_merged = merge_simple(imgs,matches_list)
plt.figure(),plt.imshow(img_merged,cmap='gray'),
plt.show()
