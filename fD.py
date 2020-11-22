################################################################################

# Alpacas & Fences - fD
# Authors: 470386390, 470354850, 470203101

# In order to run this file alone:
# $ python fD.py

# This script looks into the CV problem of finger detection.

################################################################################
# Imports
################################################################################
import cv2

################################################################################
# Main
################################################################################
# Assuming wearing long sleeve? Black background... or wearing a glove?
# Maybe have white background, wear a black glove to detect

kernel = 15
minA = 0.02
maxA = 0.9
radius = 8
sample_in = 0.02	# 2% inward from contour left and right edges
border_size = 10

# Read image data
im = cv2.imread('3.png')
width,height = im.shape[1],im.shape[0]
im_area = width * height

# def detectFinger(im,im_area):

# Preprocessing: Get cleaned binarised image with white background.
bw_im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
bw_im = cv2.GaussianBlur(bw_im,(kernel,kernel),0)
ret,thresh = cv2.threshold(bw_im,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
thresh = 255-thresh

processed_im = cv2.dilate(thresh,(kernel,kernel),iterations = 20)
# processed_im = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, (kernel,kernel))
processed_im = cv2.copyMakeBorder(processed_im,border_size,border_size,border_size,border_size,cv2.BORDER_CONSTANT,value=255)

# cv2.imshow('a',processed_im)

# # Get hand contour and find top-most point (Then change to find top most point of orientation)
contours,hier = cv2.findContours(processed_im,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(im,contours,-1,(0,255,0),3)
# print(len(contours))

for cnt in contours:
	area = cv2.contourArea(cnt)
	if area > minA * im_area and area < maxA * im_area:	# Checking if the contour is valid

		# Draw bounding rectangle around the contour and get the dimensions.
		x,y,w,h = cv2.boundingRect(cnt)
		ratio = h/w
		if ratio >= 1.5: # (or use) h >= w but ratio gives easier tuning of crossover point
			# Find topmost point (i.e. finger)
			print('Finding topmost')
			fingerLoc = tuple(cnt[cnt[:,:,1].argmin()][0])
		else:
			# In bounding box, sample number of highlighted pixels in column close to either end. Whichever has less is where the finger is.
			x,y,w,h = cv2.boundingRect(cnt)
			sub_im = processed_im[y:y+h,x:x+w]	# Crop relevant part of image then sample
			left_col = int(w*sample_in)
			right_col = int(w*(1-sample_in))
			left_samp = sum(sub_im[:,left_col])
			right_samp = sum(sub_im[:,right_col])

			if left_samp <= right_samp:	# For white hand black bg, if less white on left than right, the finger is on left
				print('Finding leftmost')
				fingerLoc = tuple(cnt[cnt[:,:,0].argmin()][0])
			else:
				print('Finding rightmost')
				fingerLoc = tuple(cnt[cnt[:,:,0].argmax()][0])

		fingerLoc = (fingerLoc[0]-border_size,fingerLoc[1]-border_size)	# Relative to main img
		im = cv2.circle(im,fingerLoc,radius,(255,255,255),-1)
		break

cv2.imshow('1',im)
cv2.waitKey(0)
