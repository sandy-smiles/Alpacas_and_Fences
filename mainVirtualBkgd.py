################################################################################

# Alpacas & Fences - mainVirtualBkgd
# Authors: 470203101, 470386390, 470354850

# In order to run this file alone:
# $ python mainVirtualBkgd.py ./virtualBkgdImgs/imageWithoutLight.PNG
# OR
# $ python mainVirtualBkgd.py ./virtualBkgdImgs/IMG20190812150924.jpg

# This script aids in the removal of the background of a person.

################################################################################
# Imports
################################################################################
import cv2
import numpy as np
from PIL import Image

from viewCamera import *
from viewFileImage import *
from imageManipulations import *
from imageClusterings import *
from imageBkgds import *

################################################################################
# Constants
################################################################################
COLOR_MAX = 255
GRAY_COLOR = 220


KERNEL = 'ellipse' # or 'box'
IMG_SCALE = 0.05 # Scaling to resize img when grouping colours together.
C_THRESH = 40 # Threshold where colour matching diff below are taken
G_THRESH = 80 # Threshold where groups above are taken

################################################################################
# Functions
################################################################################
# removeBkgd
# Replaces the background of the given bgr_img with the bkgd_img
# Input:
#   - bgr_img | cv2 image -> image object/data
# Output:
#   - bgr_img | cv2 image -> image object/data
def removeBkgd(bgr_img, show_img=0):
  gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

  frameName = 'removeBkgd'
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

  return new_bgr_img

#-------------------------------------------------------------------------------
# replaceBkgd
# Replaces the background of the given bgr_img with the bkgd_img
# Input:
#   - bgr_img | cv2 image -> image object/data
# Output:
#   - bgr_img | cv2 image -> image object/data
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

################################################################################
# Main
################################################################################
def main():
  print("[mainVirtualBkgd - INFO]: Starting mainVirtualBkgd.py")
  import sys, time

  frameName = f"Showing webcam"

  print("[mainVirtualBkgd - HELP]: To finish the program, please close the images.")

  # See if we can find the image file given
  try:
    # sys.argv[0] is the script name
    imageFileName = sys.argv[1]
    print(f"[mainVirtualBkgd - INFO]: image specified as {imageFileName}")

    # Read in the wanted image
    bgr_img = readImage(imageFileName)
    print(f"[mainVirtualBkgd - DATA]: size(bgr_img) = {bgr_img.shape}")
    frameName = f"Select img"
  except:
    print(f"[mainVirtualBkgd - WARN]: No image specified...")
    print(f"[mainVirtualBkgd - INFO]: Grabbing background image from webcam...")
    print(f"[mainVirtualBkgd - INFO]: Please move away.")
    time.sleep(1.5)

    # Start up the camera
    cap = camStart()
    bgr_img = captureImg(cap)
    print(f"[mainVirtualBkgd - INFO]: Grabbed image.")
    camFinish(cap)
    print(f"[mainVirtualBkgd - INFO]: Please come back.")

  bgr_img_resized = resize2ScreenImage(bgr_img)
  viewImage(frameName+' | bgr_img', bgr_img_resized)

  # Remove the background
  scale = 0.05
  groups_img = replaceBkgd(bgr_img, 1)
  print(f"[mainVirtualBkgd - DATA]: groups_img.shape = {groups_img.shape}")
  print(f"[mainVirtualBkgd - DATA]: np.amax(groups_img) = {np.amax(groups_img)}")
  print(f"[mainVirtualBkgd - DATA]: np.amin(groups_img) = {np.amin(groups_img)}")
  groups_img = normImg(groups_img)
  groups_img_resized = resize2ScreenImage(groups_img)
  # Show image as persistant, so that we know that it has been captured.
  viewImagePersistant(f"{frameName} | groups_img | scale={scale}", groups_img_resized)

  closeAllImages()

  print("[mainVirtualBkgd - INFO]: Finished.")

if __name__ == "__main__":
  main()



