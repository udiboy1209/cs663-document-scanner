import cv2

def get_orb_features(img, num_features):
    orb = cv2.ORB_create()
    orb.setMaxFeatures(num_features)
    orb.setEdgeThreshold(1)

    kp, des = orb.detectAndCompute(img, None)
    return (kp, des)

def match_features(des1, des2, ratio):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1,des2,k=2)

    good_matches = []
    for m,n in matches:
        if m.distance < ratio*n.distance:
            good_matches.append([m])
    return good_matches

if __name__ == '__main__':
    import sys
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

    matches = match_features(des1,des2,0.5)
    img3 = np.zeros(2)
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,img3,flags=2)
    plt.imshow(img3),plt.show()
