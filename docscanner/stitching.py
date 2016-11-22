from __future__ import print_function
import numpy as np
from feature_matching import match_features, get_orb_features
from estimation import homography_ransac
from blending import multiband_blend
from numpy import linalg as lin
import cv2
from matplotlib import pyplot as plt
__all__ = ['get_connectivity_mat','merge_simple','merge_incremental']

def get_connectivity_mat(imgs, features):
    '''
    Args:
        imgs:       list of n images
        features:   list of feature tuple (kp,des) for each image
    '''

    MATCH_RATIO = 0.7
    MIN_MATCHES = 30

    N = len(imgs)
    matches_list = [[] for i in range(N)]

    for i in xrange(N):
        kpi, desi = features[i]
        for j in xrange(i+1,N):
            kpj, desj = features[j]
            matches = match_features(desi,desj,MATCH_RATIO)

            print("%d matched with %d, no. of matches: %d" % (i,j, len(matches)))
            # img3 = np.zeros(2)
            # img3 = cv2.drawMatchesKnn(imgs[i],kpi,imgs[j],kpj,matches,img3,flags=2)
            # plt.imshow(img3),plt.show()

            if len(matches) > MIN_MATCHES:
                Xi = np.ones((3,len(matches)))
                Xj = np.ones((3,len(matches)))
                for k,m in enumerate(matches):
                    Xi[0:2,k] = np.matrix(kpi[m[0].queryIdx].pt)
                    Xj[0:2,k] = np.matrix(kpj[m[0].trainIdx].pt)

                Hij,inliers = homography_ransac(Xi,Xj,1000,5)
                matches_list[i].append((j,Hij,Xi[:,inliers],Xj[:,inliers]))

    return matches_list

def merge_simple(imgs,connectivity_mat,scale,k=6):
    N = len(imgs)

    sc_down = np.identity(3)
    sc_down[0,0] = scale
    sc_down[1,1] = scale

    sc_up = lin.inv(sc_down)

    # Guessing the shape for merged image
    Lx,Ly = int(imgs[0].shape[1]*3),int(imgs[0].shape[0]*1.5)
    merged_img = np.zeros((Ly,Lx))
    num_vals = np.zeros((Ly,Lx))
    H_to_0 = [None for i in xrange(N)]
    H_to_0[0] = np.identity(3) # First image will be added as is

    que = [0]
    visited = [False for i in xrange(N)]
    visited[0] = True

    warped_imgs = []

    while len(que) > 0:
        print("Queue:", que)
        i = que.pop(0)
        connected = connectivity_mat[i]
        img1 = imgs[i]

        # Homography for transform from img i to img 0
        Hinv = H_to_0[i]
        img_warped = cv2.warpPerspective(img1,np.dot(np.dot(sc_up,Hinv),sc_down),(Lx,Ly))

        # Increment the count at each pixel which was added by this step
        # num_vals[img_warped > 0] += 1

        # Do simple gain compensation
        common_region = np.logical_and(img_warped>0, merged_img>0)
        if len(np.where(common_region)[0]) > 0:
            avg_warped = np.sum(img_warped[common_region])
            avg_merged = np.sum(merged_img[common_region])

            print(avg_warped, avg_merged)

            g_warped = avg_merged/avg_warped
            img_warped = np.multiply(img_warped,g_warped)

        # # Add the warped img to the whole img
        img_warped[common_region] = 0
        merged_img = np.add(merged_img,img_warped)
        warped_imgs.append(img_warped)

        visited[i] = True

        # Queue all the connected imgs if not visited
        for c in sorted(connected,key=lambda c:len(c[2]),reverse=True):
            if not visited[c[0]] and c[0] not in que:
                H = c[1]

                # Transform to 0 is "transform from i to 0" * "transform to i"
                H_to_0[c[0]] = np.dot(Hinv,lin.inv(H))
                que.append(c[0])

    # To prevent division by zero
    num_vals[np.where(num_vals == 0)] = 1

    # Take average of each pixel
    merged_img = np.divide(merged_img,num_vals)
    blended_img = multiband_blend(warped_imgs,k)

    return merged_img,blended_img

def merge_incremental(imgs, features, imgs_to_merge, scale, k_blend, merge_scale):

    MATCH_RATIO = 0.75
    MIN_MATCHES = 100

    img = imgs.pop(0)
    features.pop(0)
    idx = range(1,len(imgs)+1)

    # Guessing the shape for merged image
    Lx,Ly = int(img.shape[1]*merge_scale[0]),int(img.shape[0]*merge_scale[1])
    merged_img = np.zeros((Ly,Lx),dtype='uint8')

    # Add first img to merged_img
    merged_img[0:img.shape[0],0:img.shape[1]] = img

    H_to_0 = [np.identity(3) for i in xrange(len(imgs)+1)]

    while len(imgs) > 0:
        kpm,desm = get_orb_features(merged_img, 10000)

        # plt.imshow(merged_img,cmap='gray'),plt.show()
        for i in xrange(len(imgs)): # Loop through remaining images
            img = imgs[i]
            kpi, desi = features[i]

            matches = match_features(desm,desi, MATCH_RATIO)
            print("Matches with img %d: %d" % (idx[i],len(matches)))

            img3 = np.zeros(2)
            img3 = cv2.drawMatchesKnn(merged_img,kpm,imgs[i],kpi,matches,img3,flags=2)
            plt.imsave("output/match%d.png"%idx[i],img3,cmap='gray')

            if len(matches) > MIN_MATCHES:
                print("Merging img %d" % idx[i])
                # img3 = np.zeros(2)
                # img3 = cv2.drawMatchesKnn(merged_img,kpm,img,kpi,matches,img3,flags=2)
                # plt.imshow(img3),plt.show()

                Xm = np.ones((3,len(matches)))
                Xi = np.ones((3,len(matches)))
                for k,m in enumerate(matches):
                    Xm[0:2,k] = np.matrix(kpm[m[0].queryIdx].pt)
                    Xi[0:2,k] = np.matrix(kpi[m[0].trainIdx].pt)

                Hmi,_ = homography_ransac(Xi,Xm, 1000, 6)
                img_warped = cv2.warpPerspective(img, Hmi, (Lx,Ly))

                plt.imsave("output/warped_small%i.png"%idx[i],img_warped,cmap='gray')

                update_region = np.logical_and(img_warped>0,merged_img==0)
                merged_img[update_region] = img_warped[update_region]
                # merged_img = merged_img + img_warped

                H_to_0[idx[i]] = Hmi

                imgs.pop(i)
                features.pop(i)
                idx.pop(i)
                break

    Ly,Lx = imgs_to_merge[0].shape
    Ly,Lx = int(Ly*merge_scale[1]),int(Lx*merge_scale[0])
    merged_img = np.zeros((Ly,Lx))
    warped_imgs = []

    sc_down = np.identity(3)
    sc_down[0,0] = scale
    sc_down[1,1] = scale

    sc_up = lin.inv(sc_down)

    n = 0

    for H,img in zip(H_to_0,imgs_to_merge):
        n+=1
        img_warped = cv2.warpPerspective(img, np.dot(np.dot(sc_up,H),sc_down), (Lx,Ly))
        common_region = np.logical_and(merged_img > 0, img_warped > 0)

        if len(np.where(common_region)[0]) > 0:
            avg_warped = np.sum(img_warped[common_region])
            avg_merged = np.sum(merged_img[common_region])

            print(avg_warped, avg_merged)

            g_warped = avg_merged/avg_warped
            img_warped = np.multiply(img_warped,g_warped)

        plt.imsave("output/warped_full%d.png"%n,img_warped,cmap='gray'),plt.show()
        img_warped[common_region]=0
        warped_imgs.append(img_warped)
        merged_img = np.add(merged_img,img_warped)


    return multiband_blend(warped_imgs, k_blend)
