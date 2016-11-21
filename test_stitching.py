from docscanner.stitching import get_connectivity_mat, merge_simple, merge_incremental
from docscanner.feature_matching import get_orb_features

import sys, os
import cv2
import numpy as np
from matplotlib import pyplot as plt

img_folder = sys.argv[1]
img_files = sorted([os.path.join(img_folder,f) \
                    for f in os.listdir(img_folder) \
                    if os.path.isfile(os.path.join(img_folder,f))])
# img_files = img_files[0:2]
print(img_files)

SCALE = 0.2

imgs = [cv2.imread(f,0) for f in img_files]
imgs_bw = []
for img in imgs:
    avg = np.sum(img)/(img.shape[0]*img.shape[1])
    _,img_bw = cv2.threshold(cv2.resize(img, (0,0), fx=SCALE, fy=SCALE),
                             avg, 255, cv2.THRESH_BINARY)
    imgs_bw.append(img_bw)

features = [get_orb_features(img, 20000) for img in imgs_bw]

# matches_list = get_connectivity_mat(imgs_bw, features)
img_merged2 = merge_incremental(imgs_bw, features, imgs, SCALE, 6)
#img_merged,img_blended = merge_simple(imgs,matches_list,SCALE,3)
# img_new= np.absolute(np.subtract(img_merged,img_blended))
#plt.figure(),plt.imshow(img_merged,cmap='gray')
#plt.figure(),plt.imshow(img_blended,cmap='gray')
plt.figure(),plt.imshow(img_merged2,cmap='gray')
plt.show()
