import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img= r'C:\Users\ACER\Desktop\Image processing\picture\earncap.png'
img2 = r'C:\Users\ACER\Desktop\Image processing\picture\earncap.png'

img = cv.imread(img)
img = cv.resize(img, [350,600]) #resize to smaller
img2 = cv.imread(img2)
img2 = cv.resize(img2, [350,600])


## Sequence unpacking
height, width = img.shape[:2]
height, width = img2.shape[:2]

mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64) #background
fgdModel = np.zeros((1,65),np.float64) #foreground
rect = (10,10,width-40,height-40) #rectangle(x,y,w,h)
cv.grabCut(img,mask,rect,bgdModel,fgdModel,4,cv.GC_INIT_WITH_RECT) #no. is iteration
mask = np.where((mask==2)|(mask==0),0,1).astype("uint8") #newmask=(0pix,2pix in 0(bg))(1pix,3pix in 1(fg))
grab = img*mask[:,:,np.newaxis]

mask2 = np.zeros(img2.shape[:2],np.uint8)
cv.grabCut(img2,mask2,rect,bgdModel,fgdModel,4,cv.GC_INIT_WITH_RECT) #no. is iteration
mask2 = np.where((mask2==2)|(mask2==0),0,1).astype("uint8") #newmask=(0pix,2pix in 0(bg))(1pix,3pix in 1(fg))
grab2 = img2*mask2[:,:,np.newaxis]
no_bg = np.concatenate((grab, grab2), axis=1)

background = cv.absdiff(img,grab)
background2 = cv.absdiff(img2,grab2)
background[np.where((background > [0,0,0]).all(axis = 2))] = [255,255,255]
outremove = background + grab
background2[np.where((background2 > [0,0,0]).all(axis = 2))] = [255,255,255]
outremove2 = background2 + grab2

## Convert Color
gray = cv.cvtColor(outremove, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(outremove2, cv.COLOR_BGR2GRAY)
# no_bg_gry = np.concatenate((gray, gray2), axis=1)
# cv.imshow('Original', no_bg_gry)

## Blur
img_blur = cv.GaussianBlur(gray, (5,5), 0) 
img_blur2 = cv.GaussianBlur(gray2, (5,5), 0)
# blur = np.concatenate((img_blur, img_blur2), axis=1)
# cv.imshow('Blur', blur)

### Edge Detection
img_canny = cv.Canny(img_blur, 20, 10)
img_canny2 = cv.Canny(img_blur2, 20, 10)
# canny = np.concatenate((img_canny, img_canny2), axis=1)
# cv.imshow('Canny', canny)

## Sharpening
# kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharp55 = np.array([
    [-1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1],
    [-1, -1, 9, -1, -1],
    [-1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1]
])
img_sharp = cv.filter2D(src=img_canny, ddepth=-1, kernel=sharp55)
img_sharp2 = cv.filter2D(src=img_canny2, ddepth=-1, kernel=sharp55)
# # sharp = np.concatenate((img_sharp, img_sharp2), axis=1)
# # cv.imshow('Sharp', sharp)

# ## Binary
ret, thresh = cv.threshold(img_sharp, 127, 255, cv.THRESH_BINARY)
ret2, thresh2 = cv.threshold(img_sharp2, 127, 255, cv.THRESH_BINARY)
# # binary = np.concatenate((thresh, thresh2), axis=1)
# # cv.imshow('Binary', binary)

# ## Thinning
thin = cv.ximgproc.thinning(thresh)
thin2 = cv.ximgproc.thinning(thresh2)
# thinning = np.concatenate((thin, thin2), axis=1)
# # cv.imshow('Thinning', thinning)

# ## Keypoint
# orb = cv.ORB_create(nfeatures=1000)
# kp, des = orb.detectAndCompute(thin, None)
# kp2, des2 = orb.detectAndCompute(thin2, None)

sift = cv.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(thin, None) #img1
kp2, des2 = sift.detectAndCompute(thin2, None) #img2

keypoint = cv.drawKeypoints(thin, kp, None, color=(0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
keypoint2 = cv.drawKeypoints(thin2, kp, None, color=(0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# # draw_kp = np.concatenate((keypoint, keypoint2), axis=1)
# # cv.imshow('Thinning', draw_kp)
print("\nNumber of image 1 keypoints Detected: ", len(kp))
print("\nNumber of image 2 keypoints Detected: ", len(kp2))

# ## Match
# matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_HAMMING) #Use with ORB
matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)  #Use with SIFT
nn_match = matcher.knnMatch(des, des2, 2)
bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)
matches = sorted(bf.match(des, des2), key= lambda match:match.distance)

good = []
match_ratio = 0.8 #nearest neighbor matching ratio
for m,n in nn_match:
        if m.distance < match_ratio * n.distance:
            good.append(m)
keypoints = 0
if len(kp) <= len(kp2):
    keypoints = len(kp)
else:
    keypoints = len(kp2)
print("\nGood Matches:", len(good))
print("\nGood Matches percentage: ", len(good) / keypoints * 100, "%")

# ## Draw matches
out = cv.drawMatches(thin, kp, thin2, kp2, matches, flags=2, outImg=None)
cv.imshow("output", out)

score = 0
for match in matches:
    score += match.distance
score_threshold = 200
if score/len(nn_match) < score_threshold:
    print("Fingerprints match.")
else:
    print("Fingerprints don't match.")

print(score)
print(len(nn_match))

cv.waitKey()