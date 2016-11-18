from __future__ import print_function
import numpy as np
from feature_matching import match_features, get_orb_features
from estimation import homography_ransac
from numpy import linalg as lin
from scipy.ndimage.filters import gaussian_filter
import cv2
from matplotlib import pyplot as plt
from math import sqrt

__all__ = ['get_connectivity_mat','merge_simple','merge_incremental']

def get_connectivity_mat(imgs, features):
    '''
    Args:
        imgs:       list of n images
        features:   list of feature tuple (kp,des) for each image
    '''

    MATCH_RATIO = 0.65
    NUM_MATCHES_THRES = 70

    N = len(imgs)
    H = [[] for i in range(N)]
    inlier_idx = [[] for i in range(N)]

    for i in xrange(N):
        kpi, desi = features[i]
        for j in xrange(i+1,N):
            kpj, desj = features[j]
            matches = match_features(desi,desj,MATCH_RATIO)

            if len(matches) >= NUM_MATCHES_THRES:
                # Matches are high enough, there are some common regions

                print("%d matched with %d" % (i,j))

                Xi = np.ones((3,len(matches)))
                Xj = np.ones((3,len(matches)))
                for k,m in enumerate(matches):
                    Xi[0:2,k] = np.matrix(kpi[m[0].queryIdx].pt)
                    Xj[0:2,k] = np.matrix(kpj[m[0].trainIdx].pt)

                Hij,inliers = homography_ransac(Xi,Xj,1000,5)
                H[i].append((j,Hij))
                inlier_idx[i].append((j,inliers))
                H[j].append((i,lin.inv(Hij)))
                inlier_idx[j].append((i,inliers))

    return H,inlier_idx

def multiband_blend(imgs, K, s):
    im_tmp = [i.copy() for i in imgs]
    W = []
    B = []
    for i in imgs:
        B.append(np.zeros(i.shape))
        t = np.zeros(i.shape)
        t[i>0] = 1
        W.append(t)

    for k in xrange(1,K+1):
        sk = sqrt(2*k+1)*s

        print("Iteration k=%d" % k)
        for i,im in enumerate(im_tmp):
            W[i] = gaussian_filter(W[i],sk)
            B[i] = im - gaussian_filter(im,sk)
            im = im - B[i]
            print("Image %d" % i)

    blended = np.sum([np.multiply(B[i],W[i]) for i in xrange(len(im_tmp))],axis=0)\
                /np.sum(W,axis=0)

    print(blended)
    return blended.astype('uint8')

def merge_simple(imgs,connectivity_mat):
    N = len(imgs)

    # Guessing the shape for merged image
    Lx,Ly = imgs[0].shape[1]*4,imgs[0].shape[0]*2
    merged_img = np.zeros((Ly,Lx))
    num_vals = np.zeros((Ly,Lx))
    H_to_0 = [None for i in xrange(N)]
    H_to_0[0] = np.identity(3) # First image will be added as is

    que = [0]
    visited = [False for i in xrange(N)]
    visited[0] = True

    while len(que) > 0:
        print(que)
        i = que.pop(0)
        connected = connectivity_mat[i]
        img1 = imgs[i]

        # Homography for transform from img i to img 0
        Hinv = H_to_0[i]
        img_warped = cv2.warpPerspective(img1,Hinv,(Lx,Ly))

        # Increment the count at each pixel which was added by this step
        num_vals[img_warped > 0] += 1

        # Add the warped img to the whole img
        merged_img = np.add(merged_img,img_warped)

        visited[i] = True

        # Queue all the connected imgs if not visited
        for c in connected:
            if not visited[c[0]] and c[0] not in que:
                H = c[1]

                # Transform to 0 is "transform from i to 0" * "transform to i"
                H_to_0[c[0]] = np.dot(Hinv,lin.inv(H))
                que.append(c[0])

    # To prevent division by zero
    num_vals[np.where(num_vals == 0)] = 1

    # Take average of each pixel
    merged_img = np.divide(merged_img,num_vals)

    return merged_img

def merge_incremental(imgs, features):
    img = imgs.pop(0)
    features.pop(0)
    idx = range(1,8)

    # Guessing the shape for merged image
    Lx,Ly = img.shape[1]*2,img.shape[0]*2
    merged_img = np.zeros((Ly,Lx),dtype='uint8')

    # Add first img to merged_img
    merged_img[0:img.shape[0],0:img.shape[1]] = img

    warped_imgs = [merged_img.copy()]

    while len(imgs) > 0:
        kpm,desm = get_orb_features(merged_img, 10000)

        # plt.imshow(merged_img,cmap='gray'),plt.show()
        for i in xrange(len(imgs)): # Loop through remaining images
            img = imgs[i]
            kpi, desi = features[i]

            matches = match_features(desm,desi, 0.67)
            print("Matches with img %d: %d" % (idx[i],len(matches)))

            if len(matches) > 50:
                print("Merging img %d" % idx[i])
                img3 = np.zeros(2)
                img3 = cv2.drawMatchesKnn(merged_img,kpm,img,kpi,matches,img3,flags=2)
                plt.imshow(img3),plt.show()

                Xm = np.ones((3,len(matches)))
                Xi = np.ones((3,len(matches)))
                for k,m in enumerate(matches):
                    Xm[0:2,k] = np.matrix(kpm[m[0].queryIdx].pt)
                    Xi[0:2,k] = np.matrix(kpi[m[0].trainIdx].pt)

                Hmi,_ = homography_ransac(Xi,Xm, 1000, 6)
                img_warped = cv2.warpPerspective(img, Hmi, (Lx,Ly))

                update_region = np.logical_and(img_warped>0,merged_img==0)
                merged_img[update_region] = img_warped[update_region]
                # merged_img = merged_img + img_warped

                warped_imgs.append(img_warped)

                imgs.pop(i)
                features.pop(i)
                idx.pop(i)
                break

    # avg_img = np.zeros((Ly,Lx),dtype='uint16')
    # num_vals = np.zeros((Ly,Lx),dtype='uint16')

    # for w in warped_imgs:
    #     avg_img = np.add(avg_img,w)
    #     num_vals[w>0] += 1

    # num_vals[num_vals==0] = 1
    # merged_img = avg_img/num_vals

    merged_img = multiband_blend(warped_imgs,3,10)

    return merged_img
