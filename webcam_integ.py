
import cv2
import numpy as np

kernel = 15
minA = 0.02
maxA = 0.9
radius = 8
sample_in = 0.02	# 2% inward from contour left and right edges
border_size = 10

def checkBlank(im):

	threshold = 8

	h,w,d = im.shape[0],im.shape[1],im.shape[2]

	# Divide into rows. Calculate the mean pixel value. If different, if it's empty.
	num_div = 3
	delta = w//num_div
	bins = [delta*i for i in range(num_div)]

	mean_list = []

	for r in range(num_div):
		sub_im = im[bins[r]:bins[r]+delta,:,:]
		mean_list.append(np.mean(sub_im))

	if abs(mean_list[0]-mean_list[1]) > threshold or abs(mean_list[0]-mean_list[2]) > threshold:
		print('Not blank!')
		return 0
	else:
		print('Blank')
		return 1

def detectFinger(im):

	if checkBlank(sub_im) == 1:
		return (-1,-1)

	# Preprocessing: Get cleaned binarised image with white background.
	bw_im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	bw_im = cv2.GaussianBlur(bw_im,(kernel,kernel),0)
	ret,thresh = cv2.threshold(bw_im,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	thresh = 255-thresh
	
	processed_im = cv2.dilate(thresh,(kernel,kernel),iterations = 20)
	processed_im = cv2.copyMakeBorder(processed_im,border_size,border_size,border_size,border_size,cv2.BORDER_CONSTANT,value=255)

	# # Get hand contour and find top-most point (Then change to find top most point of orientation)
	contours,hier = cv2.findContours(processed_im,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)

		if w < im.shape[1]:

			finger_loc = tuple(cnt[cnt[:,:,1].argmin()][0])
			finger_loc = (finger_loc[0]-border_size,finger_loc[1]-border_size)	# Relative to main img
			return finger_loc

	return (-1,-1)

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

finger_loc_list = []
while rval:
	rval, frame = vc.read()
	flip_horiz = cv2.flip(frame, 1)

	sub_im = flip_horiz[y1:y2,x1:x2]
	finger_loc = detectFinger(sub_im)

	if finger_loc != (-1,-1):
		finger_loc = (finger_loc[0]+x1,finger_loc[1])
		finger_loc_list.append(finger_loc)
		flip_horiz = cv2.circle(flip_horiz,finger_loc,radius,(255,0,0),-1)

	cv2.rectangle(flip_horiz,(x1,y1),(x2,y2),color=(0,0,0,0))
	cv2.imshow("preview", flip_horiz)
	key = cv2.waitKey(20)
	if key == 27: # exit on ESC
		break

vc.release()
cv2.destroyWindow("preview") 

print(finger_loc_list)
