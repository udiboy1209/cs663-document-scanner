from docscanner.stitching import get_connectivity_mat, merge_simple, \
                                 bundle_adjust
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

# merged_img = merge_incremental(imgs,features)

matches_list = get_connectivity_mat(imgs, features)
H_adjusted = bundle_adjust(matches_list)

horg = []
for ml in matches_list:
    for m in ml:
        horg.append(m[1])
hadj = []
for ml in H_adjusted:
    for m in ml:
        hadj.append(m[1])

print(horg)
print(hadj)
print([a-b for a,b in zip(horg,hadj)])

img_merged = merge_simple(imgs,H_adjusted)
img_merged2 = merge_simple(imgs,matches_list)
plt.figure(),plt.imshow(img_merged,cmap='gray'),
plt.figure(),plt.imshow(img_merged2,cmap='gray')

plt.show()
