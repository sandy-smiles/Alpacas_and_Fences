################################################################################

# Alpacas & Fences - webcam_integ
# Authors: 470386390, 470354850, 470203101

# In order to run this file alone:
# $ python webcam_integ.py

################################################################################
# Imports
################################################################################
import cv2
import numpy as np

################################################################################
# Constants
################################################################################
kernel = 15
minA = 0.02
maxA = 0.9
radius = 8
sample_in = 0.02	# 2% inward from contour left and right edges
border_size = 10
calibrate_counts = 200

text = 'Taking calibration in: '
wait_periods = 5
wait = calibrate_counts//wait_periods

################################################################################
# Functions
################################################################################
def calibrate(x1,y1,x2,y2):

	for c in range(calibrate_counts):
		rval, frame = vc.read()
		flipHorizontal = cv2.flip(frame, 1)
		cv2.rectangle(flipHorizontal,(x1,y1),(x2,y2),color=(0,0,255,0),thickness=3)
		cv2.putText(flipHorizontal, text+str(wait_periods-c//wait), ((x1-400)//2,(y1+y2)//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255),2,cv2.LINE_AA)
		cv2.imshow("preview", flipHorizontal)
		key = cv2.waitKey(20)
		if key == 27:
			break

	ref_im = flipHorizontal[y1:y2,x1:x2]
	return ref_im

#-------------------------------------------------------------------------------
def checkBlank(ref_im,im):	# Check if current image equal to reference image!

	# im = cv2.GaussianBlur(im,(kernel,kernel),0)
	# ref_im = cv2.GaussianBlur(ref_im,(kernel,kernel),0)

	diff = abs(ref_im - im)
	if sum(diff) > 500:
		return 0
	else:
		return 1

#-------------------------------------------------------------------------------
def detectFinger(ref_im,im):
	# if checkBlank == 1:

	im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	ref_im = cv2.cvtColor(ref_im,cv2.COLOR_BGR2GRAY)

	# im = cv2.GaussianBlur(im,(kernel,kernel),0)
	# ref_im = cv2.GaussianBlur(ref_im,(kernel,kernel),0)

	diff = abs(ref_im-im)
	cv2.medianBlur(diff,kernel) 
	processed_im = cv2.dilate(diff,(kernel,kernel),iterations = 20)
	
	# # Get hand contour and find top-most point (Then change to find top most point of orientation)
	contours,hier = cv2.findContours(processed_im,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	print(len(contours))

	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)

		if w < im.shape[1]:

			fingerLoc = tuple(cnt[cnt[:,:,1].argmin()][0])
			fingerLoc = (fingerLoc[0]-border_size,fingerLoc[1]-border_size)	# Relative to main img

			cv2.drawContours(im,cnt,-1,(0,255,0),3)
			return im

		# return (-1,-1)

	return im

################################################################################
# Main
################################################################################
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
    height = frame.shape[0]
    width = frame.shape[1]
else:
    rval = False

x1 = width-height*3//5
x2 = width
y1 = 0
y2 = height*3//5
bound1 = (x1,y1)
bound2 = (x2,y2)

count = 0

ref_im = calibrate(x1,y1,x2,y2)

while rval:
	rval, frame = vc.read()
	flipHorizontal = cv2.flip(frame, 1)
	sub_im = flipHorizontal[y1:y2,x1:x2]

	flipHorizontal = detectFinger(ref_im,sub_im)

	# if fingerLoc != (-1,-1):
	# 	fingerLoc = (fingerLoc[0]+x1,fingerLoc[1])
	# 	flipHorizontal = cv2.circle(flipHorizontal,fingerLoc,radius,(255,0,0),-1)

	# cv2.rectangle(flipHorizontal,(x1,y1),(x2,y2),color=(0,0,0,0))
	cv2.imshow("preview", flipHorizontal)
	key = cv2.waitKey(20)
	if key == 27:
		break

vc.release()
cv2.destroyWindow("preview") 




