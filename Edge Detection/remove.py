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
cv.imshow("canny" ,canny)

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

plt.show()
cv.waitKey(0)
cv.destroyAllWindows()




# def bgremove1(myimage):
 
#     # Blur to image to reduce noise
#     myimage = cv.GaussianBlur(myimage,(5,5), 0)
 
#     # We bin the pixels. Result will be a value 1..5
#     bins=np.array([0,51,102,153,204,255])
#     myimage[:,:,:] = np.digitize(myimage[:,:,:],bins,right=True)*51
 
#     # Create single channel greyscale for thresholding
#     myimage_grey = cv.cvtColor(myimage, cv.COLOR_BGR2GRAY)
 
#     # Perform Otsu thresholding and extract the background.
#     # We use Binary Threshold as we want to create an all white background
#     ret,background = cv.threshold(myimage_grey,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
 
#     # Convert black and white back into 3 channel greyscale
#     background = cv.cvtColor(background, cv.COLOR_GRAY2BGR)
 
#     # Perform Otsu thresholding and extract the foreground.
#     # We use TOZERO_INV as we want to keep some details of the foregorund
#     ret,foreground = cv.threshold(myimage_grey,0,255,cv.THRESH_TOZERO_INV+cv.THRESH_OTSU)  #Currently foreground is only a mask
#     foreground = cv.bitwise_and(myimage,myimage, mask=foreground)  # Update foreground with bitwise_and to extract real foreground
 
#     # Combine the background and foreground to obtain our final image
#     finalimage = background+foreground
 
#     return finalimage

# remove = bgremove1(img)
# cv.imshow('removebg', remove)
# cv.waitKey()
