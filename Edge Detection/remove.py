import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('f1.jpeg')
# cv.imshow("input",img)

#Sequence unpacking
height, width = img.shape[:2]

#Get fg by GrabCut(rect)
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64) #background
fgdModel = np.zeros((1,65),np.float64) #foreground
rect = (10,10,width-40,height-40) #rectangle(x,y,w,h)
cv.grabCut(img,mask,rect,bgdModel,fgdModel,4,cv.GC_INIT_WITH_RECT) #no. is iteration
mask = np.where((mask==2)|(mask==0),0,1).astype("uint8") #newmask=(0pix,2pix in 0(bg))(1pix,3pix in 1(fg))
grab = img*mask[:,:,np.newaxis] 
# plt.imshow(grab),plt.show()

# Get bg from the difference of img&fg
background = cv.absdiff(img,grab)

# Change all pixels in bg (that are not black) to white
background[np.where((background > [0,0,0]).all(axis = 2))] = [255,255,255]
outremove = background + grab
# cv.imshow('remove bg', outremove )
# cv.imwrite("removebgw.jpg",outremove)

#sharpen by kernel
kernel = np.array([[0,-1,0], [-1, 5,-1], [0,-1,0]])
sharp = cv.filter2D(outremove, -1, kernel)
# cv.imshow("sharpen via filter",sharp)

gray = cv.cvtColor(sharp, cv.COLOR_BGR2GRAY)
# cv.imshow("gray scale", gray)

#edge detection
blur = cv.GaussianBlur(gray, (3,3), 0)
# plt.imshow(blur, cmap='gray')

# sobelx = cv.Sobel(src=blur, ddepth=cv.CV_64F, dx=1, dy=0, ksize=5) 
# filtered_image_x = cv.convertScaleAbs(sobelx)
# sobely = cv.Sobel(src=blur, ddepth=cv.CV_64F, dx=0, dy=1, ksize=5)
# filtered_image_y = cv.convertScaleAbs(sobely)
# sobelxy = cv.Sobel(src=blur, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5)
# sobel = cv.convertScaleAbs(sobelxy)

# def auto_canny(image, sigma=0.33):
# 	# compute the median of the single channel pixel intensities
# 	v = np.median(image)
# 	# apply automatic Canny edge detection using the computed median
# 	lower = int(max(0, (1.0 - sigma) * v))
# 	upper = int(min(255, (1.0 + sigma) * v))
# 	edged = cv.Canny(image, lower, upper)
# 	return edged
# canny = auto_canny(blur)

# threshold, _ = cv.threshold(blur, 0, 255, cv.THRESH_TRIANGLE)
# def get_range(threshold, sigma=0.33):
#     thresh1 = (1-sigma) * threshold
#     thresh2 = (1+sigma) * threshold
#     return thresh1, thresh2
# outthresh = get_range(threshold)
# print(outthresh)
# canny = cv.Canny(blur, threshold1 = 169.51, threshold2 = 336.49)
# cv.imshow("canny",canny)


canny = cv.Canny(blur, threshold1 = 20, threshold2 = 10)
# cv.imshow("canny" ,canny)

# laplacian = cv.Laplacian(src=blur, ddepth=cv.CV_64F, ksize=5)
# laplace = cv.convertScaleAbs(laplacian)


# plt.subplot(221)
# plt.title("input")
# plt.imshow(gray, cmap='gray')
# plt.subplot(222)
# plt.imshow(sobel, cmap='gray')
# plt.title("sobel")
# plt.subplot(223)
# plt.imshow(canny, cmap='gray')
# plt.title("canny")
# plt.subplot(224)
# plt.imshow(laplace, cmap='gray')
# plt.title("laplacian")

#convert to binary image
ret, thresh = cv.threshold(canny, 240, 255, cv.THRESH_BINARY)
# cv.imshow('binary', thresh)

#thinning
thin = cv.ximgproc.thinning(thresh)
# cv.imshow('thinning',thin)

#keypoint
orb = cv.ORB_create(nfeatures = 20)
kp, des = orb.detectAndCompute(thin, None)
kp2, des2 = orb.detectAndCompute(thin, None)
# sift = cv.xfeatures2d.SIFT_create()
# kp = sift.detect(thin, None)
keypoint = cv.drawKeypoints(thin, kp, None, color=(0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
print("\nNumber of keypoints Detected: ", len(kp))
print("\nNumber of keypoints Detected: ", len(kp2))
# cv.imshow('keypoint', keypoint)

###matches
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)
match = bf.match(des,des2)
match = sorted(match, key=lambda x:x.distance)
out = cv.drawMatches(thin, kp, thin, kp2, match[:10], None, flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

#good points
good_points = []
low_distance = 0.6
for m,n in out:
    if m.distance < low_distance*n.distance:
        good_points.append(m)
print("good matches:",len(good_points))

out = cv.drawMatches(thin, kp, thin, kp2, good_points, None)
cv.imshow("output", out)
# # Define how similar they are
# number_keypoints = 0
# if len(kp) <= len(kp2):
#     number_keypoints = len(kp)
# else:
#     number_keypoints = len(kp2)
# print("Keypoints 1ST Image: " + str(len(kp)))
# print("Keypoints 2ND Image: " + str(len(kp2)))
# print("GOOD Matches:", len(good_points))
# print("How good it's the match: ", len(good_points) / number_keypoints * 100, "%")

plt.subplot(131)
plt.title("input")
plt.imshow(gray, cmap='gray')
plt.subplot(132)
plt.title("keypoints")
plt.imshow(keypoint, cmap='gray')
plt.subplot(133)
plt.title("match")
plt.imshow(out, cmap='gray')

plt.show()
cv.waitKey(0)
cv.destroyAllWindows()