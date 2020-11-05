
import cv2

kernel = 15
minA = 0.02
maxA = 0.9
radius = 8
sample_in = 0.02	# 2% inward from contour left and right edges
border_size = 10

def detectFinger(im):

	# Preprocessing: Get cleaned binarised image with white background.
	bw_im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	bw_im = cv2.GaussianBlur(bw_im,(kernel,kernel),0)
	ret,thresh = cv2.threshold(bw_im,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	thresh = 255-thresh
	
	processed_im = cv2.dilate(thresh,(kernel,kernel),iterations = 20)
	# processed_im = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, (kernel,kernel))
	processed_im = cv2.copyMakeBorder(processed_im,border_size,border_size,border_size,border_size,cv2.BORDER_CONSTANT,value=255)

	# # Get hand contour and find top-most point (Then change to find top most point of orientation)
	contours,hier = cv2.findContours(processed_im,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)

		if w < im.shape[1]:

			fingerLoc = tuple(cnt[cnt[:,:,1].argmin()][0])
			fingerLoc = (fingerLoc[0]-border_size,fingerLoc[1]-border_size)	# Relative to main img
			im = cv2.circle(im,fingerLoc,radius,(255,0,0),-1)
			# cv2.drawContours(im,cnt,-1,(0,255,0),3)
			return im
	return im

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
    height = frame.shape[0]
    width = frame.shape[1]
else:
    rval = False

bound1 = (width-height*3//4,0)
bound2 = (width,height*3//4)

while rval:
    rval, frame = vc.read()
    flipHorizontal = cv2.flip(frame, 1)
    sub_im = flipHorizontal[bound1[1]:bound2[1],bound1[0]:bound2[0],:]
    im_w_dot = detectFinger(sub_im)
    flipHorizontal[bound1[1]:bound2[1],bound1[0]:bound2[0],:] = im_w_dot
    cv2.rectangle(flipHorizontal,bound1,bound2,color=(0,0,0,0))
    cv2.imshow("preview", flipHorizontal)
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

vc.release()
cv2.destroyWindow("preview") 



