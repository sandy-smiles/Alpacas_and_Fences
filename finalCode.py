################################################################################

# Alpacas & Fences - finalCode
# Authors: 470354850, 470386390, 470203101

# In order to run this file alone:
# $ python finalCode.py

# This script is the main and final code submission of Alpacas and Fences demo.

################################################################################
# Imports
################################################################################
import cv2
import numpy as np
import matplotlib.pyplot as plt
import idx2numpy
import math

from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split

from skimage.feature import hog
import pickle

from PIL import Image

from viewCamera import *
from viewFileImage import *
from imageManipulations import *
from imageClusterings import *
from imageBkgds import *

################################################################################
# Globals
################################################################################
global lineMatrix
global scoreMatrix
global nRows
global nCols
global playerColourRGB
global playerTurn
global playerColour
global colourLabel 
global playerScore

COLOR_MAX = 255
GRAY_COLOR = 120


KERNEL = 'ellipse' # or 'box'
IMG_SCALE = 0.05 # Scaling to resize img when grouping colours together.
C_THRESH = 40 # Threshold where colour matching diff below are taken
G_THRESH = 35 # Threshold where groups above are taken

# kernel = 15
# minA = 0.02
# maxA = 0.9
radius = 8
# sample_in = 0.02  # 2% inward from contour left and right edges
# border_size = 10
# prev_point = (-1,-1)

################################################################################
# Functions
################################################################################
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
#-------------------------------------------------------------------------------
# captureImg
def captureImg(cap, show_img=0):
  # Check if the webcam is opened correctly
  if not cap.isOpened():
    raise IOError("Cannot open webcam")

  ret, frame = cap.read()
  frameResized = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)
  newFrame = replaceBkgd(frame)
  newFrame = cv2.resize(newFrame, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)

  newFrame[draw_box_y1:draw_box_y2,draw_box_x1:draw_box_x2,:] = frameResized[draw_box_y1:draw_box_y2,draw_box_x1:draw_box_x2,:]
  newFrame[flag_box_y1:flag_box_y2,flag_box_x1:flag_box_x2,:] = frameResized[flag_box_y1:flag_box_y2,flag_box_x1:flag_box_x2,:]

  return ret, newFrame

#-------------------------------------------------------------------------------
def replaceBkgd(bgr_img, show_img=0):
  gray_bkgd = np.ones(bgr_img.shape, np.uint8)*GRAY_COLOR # kernel size
  gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

  frameName = 'replaceBkgd'
  otsu_img = diffBkgdOtsuGray(bgr_img)
  if show_img:
    otsu_img_resized = resize2ScreenImage(otsu_img)
    viewImage(frameName+' | otsu_img', otsu_img_resized)

  # close the thresh img
  k = np.ones((50,50), np.uint8) # kernel size
  if KERNEL == 'ellipse':
    k_size = 55 # kernel size
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size)) 
  closed_otsu_img = cv2.morphologyEx(otsu_img, cv2.MORPH_CLOSE, k)
  if show_img:
    closed_otsu_img_resized = resize2ScreenImage(closed_otsu_img)
    viewImage(frameName+' | closed_otsu_img', closed_otsu_img_resized)

  scale = IMG_SCALE
  c_thresh = C_THRESH
  groups_img = createGroupsGray(resizeImage(closed_otsu_img, scale), c_thresh)
  h, w = closed_otsu_img.shape
  groups_img = cv2.resize(groups_img, (w, h), interpolation = cv2.INTER_AREA)
  groups_img = normImg(groups_img)*255
  groups_img = groups_img.astype(np.uint8)
  if show_img:
    groups_img_resized = resize2ScreenImage(groups_img)
    viewImage(frameName+' | groups_img', groups_img_resized)

  # bin-threshold
  g_thresh = G_THRESH
  _, bin_groups_img = cv2.threshold(groups_img, g_thresh, COLOR_MAX, cv2.THRESH_BINARY)
  if show_img:
    bin_groups_img_resized = resize2ScreenImage(bin_groups_img)
    viewImage(frameName+' | bin_groups_img', bin_groups_img_resized)

  # Change so that we only look at top numbered groupings
  new_mask_img = cv2.bitwise_and(bin_groups_img, bin_groups_img, mask=closed_otsu_img)
  if show_img:
    new_mask_img_resized = resize2ScreenImage(new_mask_img)
    viewImage(frameName+' | new_mask_img', new_mask_img_resized)


  # multiply with original so that we can re-otsu
  new_bgr_img = cv2.bitwise_and(bgr_img, bgr_img, mask=new_mask_img)
  if show_img:
    new_bgr_img_resized = resize2ScreenImage(new_bgr_img)
    viewImage(frameName+' | new_bgr_img', new_bgr_img_resized)

  # inverse mask to create background
  new_bkgd_img = cv2.bitwise_and(gray_bkgd, gray_bkgd, mask=~new_mask_img)
  if show_img:
    new_bkgd_img_resized = resize2ScreenImage(new_bkgd_img)
    viewImage(frameName+' | new_bkgd_img', new_bkgd_img_resized)

  # add background to masked foreground
  new_img = new_bgr_img+new_bkgd_img
  if show_img:
    new_img_resized = resize2ScreenImage(new_img)
    viewImage(frameName+' | new_img', new_img_resized)

  return new_img 

#-------------------------------------------------------------------------------
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
    #print('Not blank!')
    return 0
  else:
    #print('Blank')
    return 1

#-------------------------------------------------------------------------------
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
      finger_loc = (finger_loc[0]-border_size,finger_loc[1]-border_size)  # Relative to main img
      return finger_loc

  return (-1,-1)

#-------------------------------------------------------------------------------
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

#-------------------------------------------------------------------------------
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

#-------------------------------------------------------------------------------
# Offer options 1-4.
def getNumPlayers(vc,text_display):
  region_thresh = 15
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

  rval, frame = captureImg(vc)
  while rval:
    rval, frame = captureImg(vc)

    # DETECTING FINGER
    draw_sub_im = frame[draw_box_y1:draw_box_y2,draw_box_x1:draw_box_x2]
    rel_finger_loc = detectFinger(draw_sub_im)  # Relative finger point

    # DRAWING FINGER POINT
    if rel_finger_loc != (-1,-1):
      finger_loc = (rel_finger_loc[0]+draw_box_x1,rel_finger_loc[1])
      # finger_loc = verifyPoint(finger_loc)
      frame = cv2.circle(frame,finger_loc,radius,(255,0,0),-1)
    
    cv2.rectangle(frame,(draw_box_x1,draw_box_y1),(draw_box_x2,draw_box_y2),color=(0,0,0,0))
    cv2.rectangle(frame,(flag_box_x1,flag_box_y1),(flag_box_x2,flag_box_y2),color=(0,0,0,0))

    # Intructions for user
    cv2.putText(frame,text_display,(width//5,height*4//5), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),3,cv2.LINE_AA)

    # Draw options in box
    for r in range(1,5):
      coord1 = region_bounds[r][COORD]
      coord2 = (region_bounds[r][COORD][0] + region_bounds[r][WIDTH], region_bounds[r][COORD][1] + region_bounds[r][HEIGHT])
      text_coord = (coord1[0] + 25,coord1[1] + 75)
      cv2.rectangle(frame,coord1,coord2,color=(0,0,0,0))
      cv2.putText(frame,str(r),text_coord, cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),3,cv2.LINE_AA)

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
        if getConfirmation(vc,'Lock in number of players: '+str(curr_region), 0) == True:
          return curr_region
        else:
          # Reset values and restart
          region_count = 0
          prev_region = -1
          curr_region = -1
          finger_loc = (-1,-1)

          # Load most updated image and then output.
          rval, frame = captureImg(vc)
    else:
      region_count = 0

    prev_region = curr_region
    #print(region_count)

    cv2.imshow("preview", frame)
    key = cv2.waitKey(20)
    if key == 27:
      break

#-------------------------------------------------------------------------------
def getCharacter(vc,text_display):
  char_dict = dict()
  num_chars = 0
  prev_status = False

  drawn_char = []
  rval, frame = captureImg(vc)
  while rval:
    rval, frame = captureImg(vc)

    # DETECTING FINGER
    draw_sub_im = frame[draw_box_y1:draw_box_y2,draw_box_x1:draw_box_x2]
    rel_finger_loc = detectFinger(draw_sub_im)  # Relative finger point

    # DRAWING FINGER POINT
    if rel_finger_loc != (-1,-1):
      finger_loc = (rel_finger_loc[0]+draw_box_x1,rel_finger_loc[1])
      # finger_loc = verifyPoint(finger_loc)
      frame = cv2.circle(frame,finger_loc,radius,(255,0,0),-1)

    # DETECTING FLAG (RED PAPER). Display message to show character is being read.
    flag_sub_im = frame[flag_box_y1:flag_box_y2,flag_box_x1:flag_box_x2]
    draw_char_status = checkDrawChar(flag_sub_im)
    if draw_char_status:
      string_display = 'Reading drawn character...'
    else:
      string_display = text_display


    #print(draw_char_status)


    if draw_char_status == True:
      drawn_char.append(rel_finger_loc)

    # Check for falling edge to save things
    elif prev_status == True and draw_char_status == False:
      drawn_char = removeListValues(drawn_char,(-1,-1))
      char_dict[num_chars] = drawn_char

      letter = convertToLetter(char_dict)
      letterIndex = playerColour.index(letter)

      if getConfirmation(vc,'Lock in the colour: ' + colourLabel[letterIndex], letterIndex) == True:
        return letter
      else:
        # Reset values and restart
        # draw_char_status = False
        prev_status = False
        drawn_char = []
        char_dict = dict()

        # Load most updated image and then output.
        rval, frame = captureImg(vc)

    prev_status = draw_char_status

    cv2.rectangle(frame,(draw_box_x1,draw_box_y1),(draw_box_x2,draw_box_y2),color=(0,0,0,0))
    cv2.rectangle(frame,(flag_box_x1,flag_box_y1),(flag_box_x2,flag_box_y2),color=(0,0,0,0))

    cv2.putText(frame,string_display,(width//5,height*4//5), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),3,cv2.LINE_AA)

    cv2.imshow("preview", frame)
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
      break

#-------------------------------------------------------------------------------
def getConfirmation(vc,text_display, letterIndex):
  region_thresh = 15
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

  rval, frame = captureImg(vc)
  while rval:
    rval, frame = captureImg(vc)

    # DETECTING FINGER
    draw_sub_im = frame[draw_box_y1:draw_box_y2,draw_box_x1:draw_box_x2]
    rel_finger_loc = detectFinger(draw_sub_im)  # Relative finger point

    # DRAWING FINGER POINT
    if rel_finger_loc != (-1,-1):
      finger_loc = (rel_finger_loc[0]+draw_box_x1,rel_finger_loc[1])
      # finger_loc = verifyPoint(finger_loc)
      frame = cv2.circle(frame,finger_loc,radius,(255,0,0),-1)
    
    cv2.rectangle(frame,(draw_box_x1,draw_box_y1),(draw_box_x2,draw_box_y2),color=(0,0,0,0))
    cv2.rectangle(frame,(flag_box_x1,flag_box_y1),(flag_box_x2,flag_box_y2),color=(0,0,0,0))

    # Intructions for user
    cv2.putText(frame,text_display,(width//5,height*4//5), cv2.FONT_HERSHEY_SIMPLEX, 2, playerColourRGB[letterIndex],3,cv2.LINE_AA)

    # Draw options in box
    for r in [1,0]:
      coord1 = region_bounds[r][COORD]
      coord2 = (region_bounds[r][COORD][0] + region_bounds[r][WIDTH], region_bounds[r][COORD][1] + region_bounds[r][HEIGHT])
      text_coord = (coord1[0] + 25,coord1[1] + 75)
      cv2.rectangle(frame,coord1,coord2,color=(0,0,0,0))
      cv2.putText(frame,region_label[r],text_coord, cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),3,cv2.LINE_AA)

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
    #print(region_count)

    cv2.imshow("preview", frame)
    key = cv2.waitKey(20)
    if key == 27:
      break

#-------------------------------------------------------------------------------
# drawGrid
#def drawGrid(frame, dotsRows, dotsCols):

#  for a in range(dotsRows) :

#    for b in range(dotsCols) :
#      #image = cv2.circle(image, center_coordinates, radius, color, thickness)
#      frame = cv2.circle(frame, (round(100+b*(800/dotsCols)), round(100+a*(800/dotsRows))), 20, (0, 0, 0), 5)

#-------------------------------------------------------------------------------
# drawLine
def drawLines(frame, dotsRows, dotsCols):

  # Drawing in image lines
  for a in range(dotsRows) :
    for b in range(dotsCols) :
      for c in range(2) :
        
        if lineMatrix[a][b][c] != 0 :

          xAssign1 = b
          yAssign1 = a

          dotPoint1 = (xAssign1,yAssign1)
          coord1 = convertDot2Coord(dotPoint1)

          colourIndex = int(lineMatrix[a][b][c] - 1)
          if c == 0:
            xAssign2 = b+1
            yAssign2 = a

            dotPoint2 = (xAssign2,yAssign2)
            coord2 = convertDot2Coord(dotPoint2)
            #frame = cv2.line(frame, ( round(100+b*(800/dotsCols)), round(100+a*(800/dotsRows)) ), ( round(100+(b+1)*(800/dotsCols)), round(100+a*(800/dotsRows)) ), playerColourRGB[playerColour.index(assigned_players[colourIndex])], 5)
            frame = cv2.line(frame, coord1, coord2, playerColourRGB[playerColour.index(assigned_players[colourIndex])], 5)

          elif c == 1:
            xAssign2 = b
            yAssign2 = a+1

            dotPoint2 = (xAssign2,yAssign2)
            coord2 = convertDot2Coord(dotPoint2)
            #frame = cv2.line(frame, ( round(100+b*(800/dotsCols)), round(100+a*(800/dotsRows)) ), ( round(100+b*(800/dotsCols)), round(100+(a+1)*(800/dotsRows)) ), playerColourRGB[playerColour.index(assigned_players[colourIndex])], 5)
            frame = cv2.line(frame, coord1, coord2, playerColourRGB[playerColour.index(assigned_players[colourIndex])], 5)

        
  # Drawing in the text for a complete square
  for a in range(dotsRows-1) :
    for b in range(dotsCols-1) :
      if scoreMatrix[a][b] != 0 :
            xAssign = b
            yAssign = a
            dotPointScore = (xAssign,yAssign)
            coordScore = convertDot2Coord(dotPointScore)

            colourIndex = int(scoreMatrix[a][b] - 1)
            frame = cv2.putText(frame, str(int(scoreMatrix[a][b])) , ( round(25+coordScore[0]), round(25+coordScore[1]) ), cv2.FONT_HERSHEY_SIMPLEX , 1, playerColourRGB[playerColour.index(assigned_players[colourIndex])] , 2, cv2.LINE_AA) 

#--------------------------------------------------------------------------------------
def convertToLetter(char_dict) :

    featureList = []

    filename = './ROGM_model.sav'
		
    loaded_model = pickle.load(open(filename, 'rb'))

    for x in char_dict:
        currentSet = char_dict[x]
        image = np.zeros((432,432))

        for n in currentSet:

            for i in range(-10,10) :
                for j in range(-10,10) :
                    if (n[1] + i) >= 0 and (n[1] + i) <= 431 and (n[0] + j) >= 0 and (n[0] + j) <= 431 :
                                            
                        image[n[1]+i,n[0]+j] = 1

        resizedIm = cv2.resize(image,(28,28))
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        morphedIm = cv2.dilate(resizedIm,kernel,iterations = 1)
        kernel = np.ones((2,2),np.float32)/4
        dst = cv2.filter2D(morphedIm,-1,kernel)

        hog_Feature = hog(dst, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualize=False, multichannel=False)

        featureList.append(hog_Feature)
        y_pred = loaded_model.predict(featureList)
                        

    print(y_pred)
    return y_pred[0]

#-------------------------------------------------------------------------------
# checkValidInput
def checkValidInput(x1,y1,x2,y2):
       
    validInput = 0
    distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)

    # Check to see if valid line (e.g. coordinates are in grid and adjacent)
    if x1 <= nCols and x2 <= nCols and y1 <= nRows and y2 <= nRows and distance == 1:

        # Check to see if line is free or occupied by another player
        if x1 == x2 and lineMatrix[min(y1,y2)-1,x1-1,1] < 1 :
            lineMatrix[min(y1,y2)-1,x1-1,1] = playerTurn
            validInput = 1
            #plt.plot([x1, x2], [y1, y2], playerColour[playerTurn-1], linewidth = 10)
        elif y1 == y2 and lineMatrix[y1-1,min(x1,x2)-1,0] < 1 :
            lineMatrix[y1-1,min(x1,x2)-1,0] = playerTurn
            validInput = 1
            #plt.plot([x1, x2], [y1, y2], playerColour[playerTurn-1], linewidth = 10)
        else :
            print('Line already occupied')

    else :
        print('Invalid input')

    return validInput

#-------------------------------------------------------------------------------
# updateScore
def updateScore():

    playerScored = 0

    # Check to see if complete square and increase score if it is
    for i in range(nRows) : 
        for j in range(nCols) :    
            if lineMatrix[i-1,j-1,0] > 0 and lineMatrix[i-1,j-1,1] > 0  and lineMatrix[i,j-1,0] > 0 and lineMatrix[i-1,j,1] > 0 and scoreMatrix[i-1,j-1] == 0 :
                   
                scoreMatrix[i-1,j-1] = playerTurn
                playerScored = 1 

                #plt.plot(j+0.5, i+0.5, 'x', linewidth = 10, color = playerColour[playerTurn-1])  

                playerScore[playerTurn-1] += 1

    return playerScored

#-------------------------------------------------------------------------------
def get2Points(vc):
    dot1 = get1Point(vc,'Select Point 1')
    dot2 = get1Point(vc,'Select Point 2',point1=convertDot2Coord(dot1))
    return [dot1,dot2]

#-------------------------------------------------------------------------------
def convertDot2Coord(dot):
    offset = 100
    max_w = abs(draw_box_x1-draw_box_x2)
    dotsCols = 4
    space = (max_w-offset)//dotsCols
    dot_r = 20

    x = dot[0]
    y = dot[1]

    coord = (draw_box_x1+x*space+offset+dot_r,draw_box_y1+y*space+offset+dot_r)
    return coord

#-------------------------------------------------------------------------------
def get1Point(vc,text_display,point1=None):
    region_thresh = 15
    prev_region = -1
    curr_region = -1
    region_count = 0
    finger_loc = (-1,-1)

    region_bounds = getDotBounds((draw_box_x1,draw_box_y1),abs(draw_box_x1-draw_box_x2)-100)

    # Draw options in box
    COORD = 0
    WIDTH = 1
    HEIGHT = 2

    rval, frame = captureImg(vc)
    while rval:
        rval, frame = captureImg(vc)

        # DETECTING FINGER
        draw_sub_im = frame[draw_box_y1:draw_box_y2,draw_box_x1:draw_box_x2]
        rel_finger_loc = detectFinger(draw_sub_im)    # Relative finger point

        # DRAWING FINGER POINT
        if rel_finger_loc != (-1,-1):
            finger_loc = (rel_finger_loc[0]+draw_box_x1,rel_finger_loc[1])
            frame = cv2.circle(frame,finger_loc,radius,(255,0,0),-1)
        
        # Drawing boxes and text
        cv2.rectangle(frame,(draw_box_x1,draw_box_y1),(draw_box_x2,draw_box_y2),color=playerColourRGB[playerColour.index(assigned_players[playerTurn-1])])
        cv2.rectangle(frame,(flag_box_x1,flag_box_y1),(flag_box_x2,flag_box_y2),color=playerColourRGB[playerColour.index(assigned_players[playerTurn-1])])
        cv2.putText(frame,text_display,(width//5,height*4//5), cv2.FONT_HERSHEY_SIMPLEX, 2, playerColourRGB[playerColour.index(assigned_players[playerTurn-1])],3,cv2.LINE_AA)
        cv2.putText(frame, 'Player ' + str(playerTurn) + ' turn:' , (50, 50) , cv2.FONT_HERSHEY_SIMPLEX , 2, playerColourRGB[playerColour.index(assigned_players[playerTurn-1])] , 2, cv2.LINE_AA) 
        
        # DRAW GRID AND IDENTIFY WHERE THE FINGER IS
        frame = drawGrid(frame,region_bounds,20)
        drawLines(frame, nRows, nCols)
        for r in range(16):
            if checkIfInBound(finger_loc,region_bounds[r]):
                curr_region = r
                break
            else: curr_region = -1

        # Track finger position over time
        if curr_region == prev_region and curr_region != -1:
            region_count += 1
            if region_count > region_thresh:
                return (curr_region%4,curr_region//4)
        else:
            region_count = 0

        prev_region = curr_region


        # Draw line from current finger position to the previous selected point (if exists)
        if point1 is not None: cv2.line(frame,point1,finger_loc,(0,0,0),3)

        cv2.imshow("preview", frame)
        key = cv2.waitKey(20)
        if key == 27:
            break

#-------------------------------------------------------------------------------
def drawGrid(frame,region_bounds,dot_r):
    for i in range(len(region_bounds)):
        coord = (region_bounds[i][0],region_bounds[i][0])
        coord = (coord[0][0]+dot_r,coord[0][1]+dot_r)
        frame = cv2.circle(frame, coord, dot_r, (0, 0, 0), 3)
    return frame

#-------------------------------------------------------------------------------
def getDotBounds(ref_coord, max_w):
    ref_x = ref_coord[0]
    ref_y = ref_coord[1]
    offset = 100
    dot_r = 20
    region_bounds = dict()

    i = 0
    for a in range(nRows):
        for b in range(nCols):
            dot_coord_x = ref_x+offset+b*(max_w//nCols)
            dot_coord_y = ref_y+offset+a*(max_w//nRows)
            region_bounds[i] = [(dot_coord_x,dot_coord_y),dot_r*2,dot_r*2]
            i += 1
    return region_bounds

################################################################################
# Main
################################################################################
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

playerColour = [18, 15, 7, 13]
colourLabel = ['Red', 'Orange', 'Green', 'Magenta']
playerColourRGB = [(0, 0, 255), (0,165,255), (0,128,0), (255,0,255)]

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
    frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)
    height = frame.shape[0]
    width = frame.shape[1]
else:
    rval = False

radius = 8

draw_box_x1 = width-height*2//5
draw_box_x2 = width
draw_box_y1 = 0
draw_box_y2 = height*2//5

flag_box_x1 = 0
flag_box_x2 = height//5
flag_box_y1 = 0
flag_box_y2 = height//5

num_players = getNumPlayers(vc,'Select number of players')

assigned_players = np.zeros(num_players)
num = 0


while num < num_players :
    letter = getCharacter(vc,'Input colour for Player ' + str(num+1) )
    if letter in assigned_players : 
        print('Choose another letter')
    else:
        assigned_players[num] = letter
        num += 1


nRows = 4
nCols = 4

lineMatrix = np.zeros((nRows,nCols,2))
scoreMatrix = np.zeros((nRows-1,nCols-1))
playerScore = np.zeros(num_players)

playerTurn = 1
playerScored = 0

#playerColour = ['r', 'o', 'g', 'm']
#playerColour = [18, 15, 7, 13]

inputTracker = 0
validInput = 0

print("Player " + str(playerTurn) + " turn:")

while True:
    #rval, frame = captureImg(vc)

    # Drawing all the lines on the grid
    #drawGrid(frame, nRows, nCols)
    #drawLines(frame, nRows, nCols)

    # Text showing player and displaying current image
    #frame = cv2.putText(frame, str(playerTurn) + ':' , (25, 25) , cv2.FONT_HERSHEY_SIMPLEX , 1, playerColourRGB[playerColour.index(assigned_players[playerTurn-1])] , 2, cv2.LINE_AA) 
    #cv2.imshow('preview', frame)
    #c = cv2.waitKey(1)

    [dot1,dot2] = get2Points(vc)

    x1 = dot1[0] + 1
    y1 = dot1[1] + 1
    x2 = dot2[0] + 1
    y2 = dot2[1] + 1

    #drawLines(frame, nRows, nCols)

    validInput = checkValidInput(x1,y1,x2,y2)

    if validInput == 1 :
        playerScored = updateScore()

        # Current Player 
        if playerScored == 0:
            playerTurn = playerTurn + 1
        else :
            playerScored = 0

        if playerTurn > num_players :
            playerTurn = 1

        print("Player " + str(playerTurn) + " turn:")
        validInput = 0

vc.release()
cv2.destroyWindow('preview') 
