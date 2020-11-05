
import cv2

# Assuming wearing long sleeve? Black background... or wearing a glove?
# Convex hull
"""
Hand detection? --> Calibration and template match

Finger tip detection!
To find orientation...
Draw a bounded rectangle and look at ratio
	If height > width, get top most point
	If height < width, get left or right most point
		To check which one: see which end the convex defects of the hull are on and get there.
"""

X = 0
Y = 1
W = 2
H = 3
kernel = 5*3
minA = 0.02
maxA = 0.9
box = [142,117,118,165]
radius = 8
border_size = 10

# Read image
im = cv2.imread('test.jpg')
dim = im.shape
width = dim[1]
height = dim[0]
im_area = width * height

# Preprocessing
bw_im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
bw_im = cv2.GaussianBlur(bw_im,(kernel,kernel),0)
crop_im = bw_im[box[Y]:box[Y]+box[H],box[X]:box[X]+box[W]]
# crop_im = bw_im	# For testing

# Get binary image of hand
ret,thresh = cv2.threshold(crop_im,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
thresh = 255-thresh
dilation = cv2.dilate(thresh,(kernel,kernel),iterations = 20)
dilation = 255-dilation

# Resize img and add white border for contour detection
dilation = cv2.copyMakeBorder(dilation,border_size,border_size,border_size,border_size,cv2.BORDER_CONSTANT,value=255)

# Get hand contour and find top-most point (Then change to find top most point of orientation)
contours,hier = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(im,contours,-1,(0,255,0),3)

for cnt in contours[1:]:
	area = cv2.contourArea(cnt)
	if area > minA * im_area and area < maxA * im_area:

		# Find topmost point (i.e. finger)
		fingerLoc = tuple(cnt[cnt[:,:,1].argmin()][0])
		fingerLoc = (fingerLoc[0]+box[X]-border_size,fingerLoc[1]+box[Y]-border_size)	# Relative to main img
		
		# cv2.drawContours(im,cnt,-1,(0,255,0),3)
		im = cv2.circle(im,fingerLoc,radius,(255,255,255),-1)
		break


cv2.imshow('1',im)
cv2.waitKey(0)

# Use Template matching function later