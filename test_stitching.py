from docscanner.stitching import get_connectivity_mat, merge_incremental
from docscanner.feature_matching import get_orb_features

import sys, os
import cv2
from matplotlib import pyplot as plt

img_folder = sys.argv[1]
img_files = sorted([os.path.join(img_folder,f) \
                    for f in os.listdir(img_folder) \
                    if os.path.isfile(os.path.join(img_folder,f))])

print(img_files)

imgs = [cv2.resize(cv2.imread(f,0), (0,0), fx=0.5, fy=0.5) for f in img_files]
features = [get_orb_features(img, 10000) for img in imgs]

# Hi,inliers = get_connectivity_mat(imgs,features)

merged_img = merge_incremental(imgs,features)

plt.imshow(merged_img,cmap='gray'),plt.show()
