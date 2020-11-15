
import cv2
import numpy as np

# kernel = 15
# minA = 0.02
# maxA = 0.9
radius = 8
# sample_in = 0.02	# 2% inward from contour left and right edges
# border_size = 10
# prev_point = (-1,-1)

# Wear black glove!

def removeListValues(givenList, val):
   return [value for value in givenList if value != val]
"""
def verifyPoint(curr_point):
	error_thresh = 10
	point_thresh = 20
	error_count = 0
	
	if prev_point == (-1,-1):
		return curr_point

	x_diff = abs(curr_point[0] - prev_point[0])
	y_diff = abs(curr_point[1] - prev_point[1])

	print(x_diff,y_diff)
	if (x_diff > point_thresh) or (y_diff > point_thresh) and error_count < error_thresh:
		error_count += 1
		return prev_point
	else:
		error_count = 0
		return curr_point
"""

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
	border_size = 10
	kernel = 15

	if checkBlank(im) == 1:
		return (-1,-1)

	# Preprocessing: Get cleaned binarised image with white background.
	bw_im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	bw_im = cv2.GaussianBlur(bw_im,(kernel,kernel),0)
	ret,thresh = cv2.threshold(bw_im,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	thresh = 255-thresh
	
	processed_im = cv2.erode(thresh,(kernel,kernel),iterations = 20)
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

def checkDrawChar(im):
	hsv_im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
	hue = hsv_im[:,:,0]
	upper = hue > 200//2
	lower = hue < 44//2
	hue = upper | lower

	h = im.shape[0]
	w = im.shape[1]

	if False in hue:#np.sum(hue)/(h*w) < 0.95:
		return 0
	else:
		return 1

def checkIfInBound(pt,bound):
	ref_coord_x = bound[0][0]
	ref_coord_y = bound[0][1]
	w = bound[1]
	h = bound[2]

	pt_x = pt[0]
	pt_y = pt[1]

	if pt_x > ref_coord_x and pt_x < ref_coord_x+w and pt_y > ref_coord_y and pt_y < ref_coord_y+h:
		return True
	else:
		return False

# Offer options 1-4.
def getNumPlayers(vc,text_display):
	region_thresh = 40
	prev_region = -1
	curr_region = -1
	region_count = 0
	finger_loc = (-1,-1)

	margin = 75
	text_margin = 25
	option_box_w = 100

	region_bounds = {1:[(draw_box_x1+margin,draw_box_y1+margin),100,100],2:[(draw_box_x2-margin-option_box_w,draw_box_y1+margin),100,100],3:[(draw_box_x1+margin,draw_box_y2-margin-option_box_w),100,100],4:[(draw_box_x2-margin-option_box_w,draw_box_y2 -margin-option_box_w),100,100]}

	# Draw options in box
	COORD = 0
	WIDTH = 1
	HEIGHT = 2

	rval, frame = vc.read()
	while rval:
		rval, frame = vc.read()
		flip_horiz = cv2.flip(frame, 1)

		# DETECTING FINGER
		draw_sub_im = flip_horiz[draw_box_y1:draw_box_y2,draw_box_x1:draw_box_x2]
		rel_finger_loc = detectFinger(draw_sub_im)	# Relative finger point

		# DRAWING FINGER POINT
		if rel_finger_loc != (-1,-1):
			finger_loc = (rel_finger_loc[0]+draw_box_x1,rel_finger_loc[1])
			# finger_loc = verifyPoint(finger_loc)
			flip_horiz = cv2.circle(flip_horiz,finger_loc,radius,(255,0,0),-1)
		
		cv2.rectangle(flip_horiz,(draw_box_x1,draw_box_y1),(draw_box_x2,draw_box_y2),color=(0,0,0,0))
		cv2.rectangle(flip_horiz,(flag_box_x1,flag_box_y1),(flag_box_x2,flag_box_y2),color=(0,0,0,0))

		# Intructions for user
		cv2.putText(flip_horiz,text_display,(width//5,height*4//5), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),3,cv2.LINE_AA)

		# Draw options in box
		for r in range(1,5):
			coord1 = region_bounds[r][COORD]
			coord2 = (region_bounds[r][COORD][0] + region_bounds[r][WIDTH], region_bounds[r][COORD][1] + region_bounds[r][HEIGHT])
			text_coord = (coord1[0] + 25,coord1[1] + 75)
			cv2.rectangle(flip_horiz,coord1,coord2,color=(0,0,0,0))
			cv2.putText(flip_horiz,str(r),text_coord, cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),3,cv2.LINE_AA)

		# Identify the region the finger is in
		for r in range(1,5):
			if checkIfInBound(finger_loc,region_bounds[r]) == True:
				curr_region = r
				break
			else:
				curr_region = -1

		# Track finger position over time
		if curr_region == prev_region and curr_region != -1:
			region_count += 1
			if region_count > region_thresh:
				if getConfirmation(vc,'Lock in number of players: '+str(curr_region)) == True:
					return curr_region
				else:
					# Reset values and restart
					region_count = 0
					prev_region = -1
					curr_region = -1
					finger_loc = (-1,-1)

					# Load most updated image and then output.
					rval, frame = vc.read()
					flip_horiz = cv2.flip(frame, 1)
		else:
			region_count = 0

		prev_region = curr_region
		print(region_count)

		cv2.imshow("preview", flip_horiz)
		key = cv2.waitKey(20)
		if key == 27:
			break

def getCharacter(vc,text_display):
	char_dict = dict()
	num_chars = 0
	prev_status = False

	drawn_char = []
	rval, frame = vc.read()
	while rval:
		rval, frame = vc.read()
		flip_horiz = cv2.flip(frame, 1)

		# DETECTING FINGER
		draw_sub_im = flip_horiz[draw_box_y1:draw_box_y2,draw_box_x1:draw_box_x2]
		rel_finger_loc = detectFinger(draw_sub_im)	# Relative finger point

		# DRAWING FINGER POINT
		if rel_finger_loc != (-1,-1):
			finger_loc = (rel_finger_loc[0]+draw_box_x1,rel_finger_loc[1])
			# finger_loc = verifyPoint(finger_loc)
			flip_horiz = cv2.circle(flip_horiz,finger_loc,radius,(255,0,0),-1)

		# DETECTING FLAG (RED PAPER). Display message to show character is being read.
		flag_sub_im = flip_horiz[flag_box_y1:flag_box_y2,flag_box_x1:flag_box_x2]
		draw_char_status = checkDrawChar(flag_sub_im)
		if draw_char_status: string_display = 'Reading drawn character...'
		else: string_display = text_display


		print(draw_char_status)


		if draw_char_status == True:
			drawn_char.append(rel_finger_loc)

		# Check for falling edge to save things
		elif prev_status == True and draw_char_status == False:
			drawn_char = removeListValues(drawn_char,(-1,-1))
			char_dict[num_chars] = drawn_char

			if getConfirmation(vc,'Lock in the drawn character?') == True:
				return char_dict
			else:
				# Reset values and restart
				# draw_char_status = False
				prev_status = False
				drawn_char = []
				char_dict = dict()

				# Load most updated image and then output.
				rval, frame = vc.read()
				flip_horiz = cv2.flip(frame, 1)

		prev_status = draw_char_status

		cv2.rectangle(flip_horiz,(draw_box_x1,draw_box_y1),(draw_box_x2,draw_box_y2),color=(0,0,0,0))
		cv2.rectangle(flip_horiz,(flag_box_x1,flag_box_y1),(flag_box_x2,flag_box_y2),color=(0,0,0,0))

		cv2.putText(flip_horiz,string_display,(width//5,height*4//5), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),3,cv2.LINE_AA)

		cv2.imshow("preview", flip_horiz)
		key = cv2.waitKey(20)
		if key == 27: # exit on ESC
			break

def getConfirmation(vc,text_display):
	region_thresh = 40
	prev_region = -1
	curr_region = -1
	region_count = 0
	finger_loc = (-1,-1)

	margin = 75
	text_margin = 25
	option_box_h = 100
	option_box_w = abs(draw_box_x1-draw_box_x2)-2*margin

	region_bounds = {1:[(draw_box_x1+margin,draw_box_y1+margin),option_box_w,option_box_h],0:[(draw_box_x1+margin,draw_box_y2-margin-option_box_h),option_box_w,option_box_h]}
	region_label = {1:'Confirm',0:'Cancel'}

	# Draw options in box
	COORD = 0
	WIDTH = 1
	HEIGHT = 2

	rval, frame = vc.read()
	while rval:
		rval, frame = vc.read()
		flip_horiz = cv2.flip(frame, 1)

		# DETECTING FINGER
		draw_sub_im = flip_horiz[draw_box_y1:draw_box_y2,draw_box_x1:draw_box_x2]
		rel_finger_loc = detectFinger(draw_sub_im)	# Relative finger point

		# DRAWING FINGER POINT
		if rel_finger_loc != (-1,-1):
			finger_loc = (rel_finger_loc[0]+draw_box_x1,rel_finger_loc[1])
			# finger_loc = verifyPoint(finger_loc)
			flip_horiz = cv2.circle(flip_horiz,finger_loc,radius,(255,0,0),-1)
		
		cv2.rectangle(flip_horiz,(draw_box_x1,draw_box_y1),(draw_box_x2,draw_box_y2),color=(0,0,0,0))
		cv2.rectangle(flip_horiz,(flag_box_x1,flag_box_y1),(flag_box_x2,flag_box_y2),color=(0,0,0,0))

		# Intructions for user
		cv2.putText(flip_horiz,text_display,(width//5,height*4//5), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),3,cv2.LINE_AA)

		# Draw options in box
		for r in [1,0]:
			coord1 = region_bounds[r][COORD]
			coord2 = (region_bounds[r][COORD][0] + region_bounds[r][WIDTH], region_bounds[r][COORD][1] + region_bounds[r][HEIGHT])
			text_coord = (coord1[0] + 25,coord1[1] + 75)
			cv2.rectangle(flip_horiz,coord1,coord2,color=(0,0,0,0))
			cv2.putText(flip_horiz,region_label[r],text_coord, cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),3,cv2.LINE_AA)

		# Identify the region the finger is in
		for r in [1,0]:
			if checkIfInBound(finger_loc,region_bounds[r]) == True:
				curr_region = r
				break
			else:
				curr_region = -1

		# Track finger position over time
		if curr_region == prev_region and curr_region != -1:
			region_count += 1
			if region_count > region_thresh:
				if curr_region == 1:
					return True
				else:
					return False
				# return curr_region
		else:
			region_count = 0

		prev_region = curr_region
		print(region_count)

		cv2.imshow("preview", flip_horiz)
		key = cv2.waitKey(20)
		if key == 27:
			break

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
    height = frame.shape[0]
    width = frame.shape[1]
else:
    rval = False

radius = 8

draw_box_x1 = width-height*3//5
draw_box_x2 = width
draw_box_y1 = 0
draw_box_y2 = height*3//5

flag_box_x1 = 0
flag_box_x2 = height//5
flag_box_y1 = 0
flag_box_y2 = height//5

num_players = getNumPlayers(vc,'Select number of players')
char_dict = getCharacter(vc,'Input colour. Awaiting Signal.')

vc.release()
cv2.destroyWindow("preview") 
