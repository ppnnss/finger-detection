import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def data(location):
    path = location
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    mask = np.zeros(img.shape[:2], dtype='uint8')
    bgd = np.zeros((1, 65), dtype='float64')
    fgd = np.zeros((1, 65), dtype='float64')
    rect = (10, 10, w-40, h-40)
    cv.grabCut(img,mask,rect,bgd,fgd,4,cv.GC_INIT_WITH_RECT)
    mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    new = mask[:,:,np.newaxis]
    remove = np.multiply(img,new)
    bg = cv.absdiff(img, remove)
    bg[np.where((bg > [0,0,0]).all(axis = 2))] = [255,255,255]
    img_nobg = bg + remove
    return img_nobg
img = data(r'C:\Users\ACER\Desktop\Image processing\picture\removebgb.jpg')
img2 = data(r'C:\Users\ACER\Desktop\Image processing\picture\removebgb.jpg')

def grayscale(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return gray
gray = grayscale(img)
gray2 = grayscale(img2)

def edge(img, k1, k2, thres1, thres2):
    kernel_size = (k1, k2)
    sharp33 = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    img_blur = cv.GaussianBlur(img, kernel_size, 0)
    img_canny = cv.Canny(img_blur, thres1, thres2)
    img_sharp = cv.filter2D(src=img_canny, ddepth=-1, kernel=sharp33)
    return img_sharp
edge_detect = edge(gray, 3, 3, 20, 10)
edge_detect2 = edge(gray2, 3, 3, 20, 10)

def run(img1, img2):
    ## Convert to Binary
    ret, thresh = cv.threshold(img1, 240, 255, cv.THRESH_BINARY)
    ret2, thresh2 = cv.threshold(img2, 240, 255, cv.THRESH_BINARY)

    ## Thinning
    thin = cv.ximgproc.thinning(thresh)
    thin2 = cv.ximgproc.thinning(thresh2)

    ## ORB Keypoint
    orb = cv.ORB_create(nfeatures=10000)
    kp, des = orb.detectAndCompute(thin, None)
    kp2, des2 = orb.detectAndCompute(thin2, None)
    cv.drawKeypoints(thin, kp, None, color=(0, 255, 0), flags=0)
    cv.drawKeypoints(thin2, kp2, None, color=(0, 255, 0), flags=0)
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_HAMMING)
    nn_match = matcher.knnMatch(des, des2, 2)
    good = []
    match_ratio = 0.8 #nearest neighbor matching ratio
    for m,n in nn_match:
            if m.distance < match_ratio*(n.distance):
                good.append(m)

    number_keypoints = 0
    if len(kp) <= len(kp2):
        number_keypoints = len(kp)
    else:
        number_keypoints = len(kp2)
    print("GOOD Matches:", len(good))
    print("How good it's the match: ", len(good) / number_keypoints * 100, "%")
    return ret, ret2

result = run(edge_detect, edge_detect2)
plt.imshow(result)
plt.show()