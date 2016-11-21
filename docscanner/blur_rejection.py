from __future__ import print_function
import cv2
import sys
from matplotlib import pyplot as plt

def variance_of_laplacian(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()


def blur_rejection(img, thresh=1):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    text = "not blurry"

    if (fm < thresh):
        text = "Blurry"

    return text,fm


if __name__ == '__main__':
    text = []
    fm = []
    for i in range(1,8):
        img = cv2.imread("../testcases/" + str(i) + ".jpg")
        text1,fm1 = blur_rejection(img)
        text.append(text1)
        fm.append(fm1)
    print (text)
    print (fm)
    # img = cv2.imread(sys.argv[1])
    # text,fm = blur_rejection(img)
    # plt.subplot(121),plt.imshow(img,cmap='gray'),plt.title("{}: {:.2f}".format(text, fm))
    # img = cv2.imread(sys.argv[2])
    # text,fm = blur_rejection(img)
    # plt.subplot(122),plt.imshow(img,cmap='gray'),plt.title("{}: {:.2f}".format(text, fm))
    # plt.show()

