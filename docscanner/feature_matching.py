import cv2

__all__ = ['get_orb_features','match_features']

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
