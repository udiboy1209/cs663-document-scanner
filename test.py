from docscanner.estimation import *
from docscanner.feature_matching import *

import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt


img1 = cv2.imread(sys.argv[1],0)
img1 = cv2.resize(img1, (0,0), fx=0.2, fy=0.2)
kp1, des1 = get_orb_features(img1, 10000)

img2 = cv2.imread(sys.argv[2],0)
img2 = cv2.resize(img2, (0,0), fx=0.2, fy=0.2)
kp2, des2 = get_orb_features(img2, 10000)

#img2 = cv2.drawKeypoints(img,kp,color=(0,255,0), flags=0)
#plt.imshow(img2),plt.show()

matches = match_features(des1,des2,0.3)
img3 = np.zeros(2)
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,img3,flags=2)
#plt.imshow(img3),plt.show()


Xi1 = np.ones((3,len(matches)))
Xi2 = np.ones((3,len(matches)))
for i,m in enumerate(matches):
    Xi1[0:2,i] = np.matrix(kp1[m[0].queryIdx].pt)
    Xi2[0:2,i] = np.matrix(kp2[m[0].trainIdx].pt)

print(Xi1)
print(Xi2)
homography_ransac(Xi1,Xi2,1,1)
