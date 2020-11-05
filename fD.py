
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
box = [0,0,0,0]
# box = [142,117,118,165]
radius = 8
border_size = 0#10
sample_in = 0.02 # i.e. 5% inward

# Read image
im = cv2.imread('test.jpg')
dim = im.shape
width = dim[1]
height = dim[0]
im_area = width * height

# Preprocessing
bw_im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
bw_im = cv2.GaussianBlur(bw_im,(kernel,kernel),0)
# crop_im = bw_im[box[Y]:box[Y]+box[H],box[X]:box[X]+box[W]]
crop_im = bw_im

# Get binary image of hand
ret,thresh = cv2.threshold(crop_im,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
thresh = 255-thresh

# dilation = cv2.dilate(thresh,(kernel,kernel),iterations = 1)
# dilation = 255-dilation

dilation = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, (kernel,kernel))


# Resize img and add white border for contour detection
# dilation = cv2.copyMakeBorder(dilation,border_size,border_size,border_size,border_size,cv2.BORDER_CONSTANT,value=255)

# Get hand contour and find top-most point (Then change to find top most point of orientation)
contours,hier = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(im,contours,-1,(0,255,0),3)
print(len(contours))

for cnt in contours:
	area = cv2.contourArea(cnt)
	if area > minA * im_area and area < maxA * im_area:

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
			sub_im = dilation[y:y+h,x:x+w]	# Crop relevant part of image then sample
			left_col = int(w*sample_in)
			right_col = int(w*(1-sample_in))
			left_samp = sum(sub_im[:,left_col])
			right_samp = sum(sub_im[:,right_col])

			if left_samp >= right_samp:	# If more white on left than right, the finger is on left
				print('Finding leftmost')
				fingerLoc = tuple(cnt[cnt[:,:,0].argmin()][0])
			else:
				print('Finding rightmost')
				fingerLoc = tuple(cnt[cnt[:,:,0].argmax()][0])

		fingerLoc = (fingerLoc[0]+box[X]-border_size,fingerLoc[1]+box[Y]-border_size)	# Relative to main img		
		# cv2.drawContours(im,cnt,-1,(0,255,0),3)
		im = cv2.circle(im,fingerLoc,radius,(255,255,255),-1)
		break

cv2.imshow('1',im)
cv2.waitKey(0)

# Use Template matching function later?
