from __future__ import print_function
import numpy as np
from numpy import linalg as lin
from math import sqrt

__all__ = ['choose_points','normalise_xi','normalised_DLT','homography_ransac']

combs_4c3 = [[0,1,2],[0,1,3],[0,2,3],[1,2,3]]

# def area_tri(p):
#     x = p[0]
#     y = p[1]

#     print(x,y)
#     a = 0.5*abs((x[1]-x[0])*(y[2]-y[0])-(y[1]-y[0])*(x[2]-x[0]))
#     return a

def measure_collinearity(p):
    d = min([abs(lin.det(p[:,k])) for k in combs_4c3])
    return d

def choose_points(points,N,thres):
    n = 0
    uniq_idx = []
    s = points.shape[1]

    while n < N:
        choice_idx = tuple(np.sort(np.random.choice(s,size=4,replace=False)))
        new_choice = points[:,choice_idx]
        d = measure_collinearity(new_choice)
        if d > thres and choice_idx not in uniq_idx:
            uniq_idx.append(choice_idx)
            n += 1

    return uniq_idx

def normalise_xi(Xi):
    '''
    Args:
        Xi (np.matrix):
    '''
    T = np.zeros((3,3))
    T[2,2] = 1

    n = Xi.shape[1]

    x_mean = np.sum(Xi[0])/n
    y_mean = np.sum(Xi[1])/n

    d_sum = np.sum(np.sqrt(np.add(
                np.square(np.subtract(Xi[0],x_mean)),
                np.square(np.subtract(Xi[1],y_mean))
            )))

    s = sqrt(2)*n/d_sum

    T[0,2] = -s*x_mean
    T[1,2] = -s*y_mean
    T[0,0] = s
    T[1,1] = s

    Xi_norm = np.dot(T,Xi)
    return Xi_norm, T

def normalised_DLT(p1, p2):
    n = p1.shape[1]

    p1_norm,T1 = normalise_xi(p1)
    p2_norm,T2 = normalise_xi(p2)

    A = np.zeros((2*n,9))

    for i in xrange(n):
        Ai = np.zeros((2,9))
        XiT = np.transpose(p1_norm[:,i])
        Ai[0,3:6] = -XiT
        Ai[1,0:3] = XiT
        Ai[0,6:9] = p2_norm[1,i]*XiT
        Ai[1,6:9] = -p2_norm[0,i]*XiT

        A[2*i:2*i+2,:] = Ai

    eigVal, eigVec = lin.eig(np.dot(np.transpose(A),A))
    min_idx = eigVal.argsort()[0]

    h_norm = eigVec[:,min_idx]
    h_norm.shape = (3,3)

    H = np.dot(lin.inv(T2),np.dot(h_norm,T1))

    return H

def homography_ransac(Xi1, Xi2, N, dist_thres):
    '''
    Args:
        Xi1:            Feature points in image 1
        Xi2:            Feature points in image 2
        N:              Number of iterations
        dist_thres:     Threshold distance for inliers
    '''
    max_inliers = -1
    min_std = 10000

    best_H = None
    best_H_inliers = None

    for idx in choose_points(Xi1,N,0.1):
        p1 = Xi1[:,idx]
        p2 = Xi2[:,idx]
        Hp = normalised_DLT(p1,p2)

        Xi2_T = np.dot(Hp,Xi1)
        Xi2_T = Xi2_T / Xi2_T[2][None,:]
        Xi1_T = np.dot(lin.inv(Hp),Xi2)
        Xi1_T = Xi1_T / Xi1_T[2][None,:]

        di = np.add(
                np.sqrt(np.sum(np.square(np.subtract(Xi1_T,Xi1)),axis=0)),
                np.sqrt(np.sum(np.square(np.subtract(Xi2_T,Xi2)),axis=0))
             )

        di_std = np.std(di)
        inlier_idx = np.where(di < dist_thres)[0]

        if inlier_idx.shape[0] > max_inliers or \
           (inlier_idx.shape[0] == max_inliers and di_std < min_std):
            best_H = Hp
            best_H_inliers = inlier_idx
            max_inliers = inlier_idx.shape[0]
            min_std = di_std

    H_final = normalised_DLT(Xi1[:,best_H_inliers],Xi2[:,best_H_inliers])
    return best_H
