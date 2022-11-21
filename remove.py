import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('earn.png')
img2 = cv.imread('earn2.png')
# cv.imshow("input",img)
# cv.imshow("input",img2)

#Sequence unpacking
height, width = img.shape[:2]
height, width = img2.shape[:2]

#Get fg by GrabCut(rect)
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64) #background
fgdModel = np.zeros((1,65),np.float64) #foreground
rect = (10,10,width-40,height-40) #rectangle(x,y,w,h)
cv.grabCut(img,mask,rect,bgdModel,fgdModel,4,cv.GC_INIT_WITH_RECT) #no. is iteration
mask = np.where((mask==2)|(mask==0),0,1).astype("uint8") #newmask=(0pix,2pix in 0(bg))(1pix,3pix in 1(fg))
grab = img*mask[:,:,np.newaxis] 
# plt.imshow(grab)
mask2 = np.zeros(img2.shape[:2],np.uint8)
cv.grabCut(img2,mask2,rect,bgdModel,fgdModel,4,cv.GC_INIT_WITH_RECT) #no. is iteration
mask2 = np.where((mask2==2)|(mask2==0),0,1).astype("uint8") #newmask=(0pix,2pix in 0(bg))(1pix,3pix in 1(fg))
grab2 = img2*mask2[:,:,np.newaxis] 
# plt.imshow(grab2)

# Get bg from the difference of img&fg
background = cv.absdiff(img,grab)
background2 = cv.absdiff(img2,grab2)

# # Change all pixels in bg (that are not black) to white
background[np.where((background > [0,0,0]).all(axis = 2))] = [255,255,255]
outremove = background + grab
# cv.imshow('remove bg', outremove )
background2[np.where((background2 > [0,0,0]).all(axis = 2))] = [255,255,255]
outremove2 = background2 + grab2
# cv.imshow('remove bg2', outremove2 )

# #sharpen by kernel
kernel = np.array([[0,-1,0], [-1, 5,-1], [0,-1,0]])
sharp = cv.filter2D(outremove, -1, kernel)
# cv.imshow("sharpen via filter",sharp)
kernel2 = np.array([[0,-1,0], [-1, 5,-1], [0,-1,0]])
sharp2 = cv.filter2D(outremove2, -1, kernel2)
# cv.imshow("sharpen via filter",sharp2)

gray = cv.cvtColor(sharp, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(sharp2, cv.COLOR_BGR2GRAY)
# cv.imshow("gray scale", gray)
# cv.imshow("gray scale", gray2)

# #edge detection
blur = cv.GaussianBlur(gray, (3,3), 0)
blur2 = cv.GaussianBlur(gray2, (3,3), 0)
# plt.imshow(blur, cmap='gray')
# plt.imshow(blur2, cmap='gray')

canny = cv.Canny(blur, threshold1 = 20, threshold2 = 10)
# cv.imshow("canny" ,canny)
canny2 = cv.Canny(blur2, threshold1 = 20, threshold2 = 10)
# cv.imshow("canny" ,canny2)

# convert to binary image
ret, thresh = cv.threshold(canny, 240, 255, cv.THRESH_BINARY)
# cv.imshow('binary', thresh)
ret2, thresh2 = cv.threshold(canny2, 240, 255, cv.THRESH_BINARY)
# cv.imshow('binary', thresh2)

#thinning
thin = cv.ximgproc.thinning(thresh)
cv.imshow('thinning',thin)
thin2 = cv.ximgproc.thinning(thresh2)
cv.imshow('thinning2',thin2)


#keypoint
orb = cv.ORB_create(nfeatures=10000) #no. of kp
kp, des = orb.detectAndCompute(thin, None) #img1
kp2, des2 = orb.detectAndCompute(thin2, None) #img2
# sift = cv.xfeatures2d.SIFT_create()
# kp = sift.detect(thin, None)
keypoint = cv.drawKeypoints(thin, kp, None, color=(0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
keypoint2 = cv.drawKeypoints(thin2, kp, None, color=(0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
print("\nNumber of keypoints Detected: ", len(kp))
print("\nNumber of keypoints Detected: ", len(kp2))
# cv.imshow('keypoint', keypoint)
# cv.imshow('keypoint2', keypoint2)

###matches
# bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)
# match = bf.match(des,des2)
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

# match = sorted(match, key=lambda x:x.distance)
# out = cv.drawMatches(thin, kp, thin, kp2, match[:10], None, flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# print(good)
# out = cv.drawMatches(thin, kp, thin, kp2, good_points, None)
# cv.imshow("output", out)


# plt.subplot(131)
# plt.title("input")
# plt.imshow(gray, cmap='gray')
# plt.subplot(132)
# plt.title("keypoints")
# plt.imshow(keypoint, cmap='gray')
# plt.subplot(133)
# plt.title("match")
# plt.imshow(out, cmap='gray')

plt.show()
cv.waitKey(0)
cv.destroyAllWindows()