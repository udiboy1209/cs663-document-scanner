import numpy as np
import cv2

def multiband_blend(imgs, K):
    # Gp_all stores all gaussian pyramids of all imgs
    Gp_all = [[i.copy().astype(float)] for i in imgs]

    print("Creating Gp")
    for i in xrange(K):
        for Gp in Gp_all:
            G = cv2.pyrDown(Gp[-1])
            Gp.append(G)

    # Stores all laplacian pyramids of all imgs
    Lp_all = [[Gp[-1]] for Gp in Gp_all]

    print("Creating Lp line 17 ", Gp_all[0][-1].shape, Gp_all[0][K].shape)
    for i in xrange(K,0,-1):
        for Lp,Gp in zip(Lp_all,Gp_all):
            print("Shapes: line LpA ",Gp[i-1].shape,Gp[i].shape)
            GE = cv2.pyrUp(Gp[i])
            xoff,yoff = GE.shape[0] - Gp[i-1].shape[0], \
                        GE.shape[1] - Gp[i-1].shape[1]
            GE = GE[xoff:,yoff:]
            print("Shapes: line LpB ",Gp[i-1].shape,GE.shape)
            L = cv2.subtract(Gp[i-1],GE)
            Lp.append(L)

    del(Gp_all)

    # Calc merged pyramids at all levels
    Ls = []
    for i in xrange(K+1):
        print("Adding level %i" % i)
        n = 1
        ls = Lp_all[0][i]
        #ls = np.empty(0)
        for Lp in Lp_all[1:]:
            ls = np.add(ls,Lp[i])
            print("Adding img %d" % n)
            n += 1

        Ls.append(ls)

    del(Lp_all)

    blended = Ls[0]
    for i in xrange(1,K+1):
        print("Blending level %i" % i)
        print("Shapes of blended: A ",blended.shape, Ls[i].shape)
        blended = cv2.pyrUp(blended)
        xoff,yoff = blended.shape[0] - Ls[i].shape[0], \
                    blended.shape[1] - Ls[i].shape[1]
        blended = blended[xoff:,yoff:]
        print("Shapes of blended: B ",blended.shape, Ls[i].shape)
        blended = cv2.add(blended, Ls[i])
    return blended
