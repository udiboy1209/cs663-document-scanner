import cv2

def get_orb_features(img, num_features):
    orb = cv2.ORB_create()
    orb.setMaxFeatures(num_features)
    orb.setEdgeThreshold(1)

    kp, des = orb.detectAndCompute(img, None)
    return (kp, des)

if __name__ == '__main__':
    import sys
    import numpy as np
    from matplotlib import pyplot as plt

    desses = []
    kps = []
    imgs = []

    for f in sys.argv[1:]:
        img = cv2.imread(f,0)
        img = cv2.resize(img, (0,0), fx=0.2, fy=0.2)
        #img = np.pad(img,50,'constant')
        kp, des = get_orb_features(img, 10000)

        desses.append(des)
        kps.append(kp)
        imgs.append(img)

        #img2 = cv2.drawKeypoints(img,kp,color=(0,255,0), flags=0)
        #plt.imshow(img2),plt.show()

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desses[0],desses[1])

    matches = [m for m in matches if m.distance < 20]
    img3 = np.zeros(2)
    img3 = cv2.drawMatches(imgs[0],kps[0],imgs[1],kps[1],matches[:50],img3,flags=2)
    plt.imshow(img3),plt.show()
