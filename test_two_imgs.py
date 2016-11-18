from docscanner.estimation import homography_ransac
from docscanner.feature_matching import get_orb_features,match_features

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

matches = match_features(des1,des2,0.65)
img3 = np.zeros(2)
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,img3,flags=2)
plt.figure(),plt.imshow(img3)

print("No. of matches: %d" % len(matches))

if len(matches) < 50:
    print("Matches too low")
    exit()

Xi1 = np.ones((3,len(matches)))
Xi2 = np.ones((3,len(matches)))
for i,m in enumerate(matches):
    Xi1[0:2,i] = np.matrix(kp1[m[0].queryIdx].pt)
    Xi2[0:2,i] = np.matrix(kp2[m[0].trainIdx].pt)

h,_ = homography_ransac(Xi1,Xi2,1000,5.5)
# h_cv,status = cv2.findHomography(np.transpose(Xi1[0:2,:]),np.transpose(Xi2[0:2,:]))

if h.any():
    print(h)

    merged_size = (img1.shape[1]*2,img1.shape[0]*2)
    img_warp = cv2.warpPerspective(img2,np.linalg.inv(h),merged_size)
    img_warp[0:img1.shape[0],0:img1.shape[1]] = img1

    #plt.figure(),plt.imshow(img2,cmap='gray'),plt.title("2")
    plt.figure(),plt.imshow(img_warp,cmap='gray'),plt.title("Warped 2")
    #plt.figure(),plt.imshow(img1,cmap='gray'),plt.title("1")
    plt.show()
else:
    print("Images don't match, couldn't find homography")
