################################################################################

# Alpacas & Fences - mainVirtualBkgd
# Authors: 470203101, 470386390, 470345850

# In order to run this file alone:
# $ python mainVirtualBkgd.py ./virtualBkgdImgs/IMG20190812150924.jpg

# This script aids in the removal of the background of a person.
# Follows the instructions from https://www.learnopencv.com/applications-of-foreground-background-separation-with-semantic-segmentation/

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

################################################################################
# Functions
################################################################################
# removeBkgd
# Replaces the background of the given bgr_img with the bkgd_img
# Input:
#   - bgr_img | cv2 image -> image object/data
# Output:
#   - N/A
def removeBkgd(bgr_img):
  gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

  frameName = 'removeBkgd'
  otsu_img = diffBkgdOtsuGray(bgr_img)
  otsu_img_resized = resize2ScreenImage(otsu_img)
  viewImage(frameName+' | otsu_img', otsu_img_resized)

  # close the thresh img
  #k = np.ones((50,50), np.uint8) # kernel size
  k_size = 55 # kernel size
  k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size)) 
  closed_otsu_img = cv2.morphologyEx(otsu_img, cv2.MORPH_CLOSE, k)
  closed_otsu_img_resized = resize2ScreenImage(closed_otsu_img)
  viewImage(frameName+' | closed_otsu_img', closed_otsu_img_resized)

  scale = 0.05
  c_thresh = 40
  groups_img = createGroupsGray(resizeImage(closed_otsu_img, scale), c_thresh)
  h, w = closed_otsu_img.shape
  groups_img = cv2.resize(groups_img, (w, h), interpolation = cv2.INTER_AREA)
  groups_img = normImg(groups_img)*255
  groups_img = groups_img.astype(np.uint8)
  groups_img_resized = resize2ScreenImage(groups_img)
  viewImage(frameName+' | groups_img', groups_img_resized)

  # bin-threshold
  g_thresh = 80
  _, bin_groups_img = cv2.threshold(groups_img, g_thresh, COLOR_MAX, cv2.THRESH_BINARY)
  bin_groups_img_resized = resize2ScreenImage(bin_groups_img)
  viewImage(frameName+' | bin_groups_img', bin_groups_img_resized)

  # Change so that we only look at top numbered groupings
  new_mask_img = cv2.bitwise_and(bin_groups_img, bin_groups_img, mask=closed_otsu_img)
  new_mask_img_resized = resize2ScreenImage(new_mask_img)
  viewImage(frameName+' | new_mask_img', new_mask_img_resized)


  # multiply with original so that we can re-otsu
  new_bgr_img = cv2.bitwise_and(bgr_img, bgr_img, mask=new_mask_img)
  new_bgr_img_resized = resize2ScreenImage(new_bgr_img)
  viewImage(frameName+' | new_bgr_img', new_bgr_img_resized)

  return new_bgr_img

#-------------------------------------------------------------------------------
# replaceBkgd
# Replaces the background of the given bgr_img with the bkgd_img
# Input:
#   - bgr_img | cv2 image -> image object/data
#   - bkgd_img | cv2 image -> image object/data
# Output:
#   - N/A
def replaceBkgd(bgr_img, bkgd_img):
  return bgr_img

################################################################################
# Main
################################################################################
def mainTakenInit():
  import sys, time
  frameName = f"Showing webcam"

  print("[mainVirtualBkgd - INFO]: Starting mainVirtualBkgd.py")
  print("[mainVirtualBkgd - HELP]: To finish the program, please close the images.")

  # See if we can find the image file given
  try:
    # sys.argv[0] is the script name
    imageFileName = sys.argv[1]

    # Read in the wanted image
    bkgd_bgr_img = readImage(imageFileName)
    print(f"[mainVirtualBkgd - DATA]: size(bkgd_bgr_img) = {bkgd_bgr_img.shape}")
  except:
    print(f"[mainVirtualBkgd - WARN]: No image specified...")
    print(f"[mainVirtualBkgd - INFO]: Grabbing background image from webcam...")
    print(f"[mainVirtualBkgd - INFO]: Please move away.")
    time.sleep(1.5)

    # Start up the camera
    cap = camStart()
    bkgd_bgr_img = captureImg(cap)
    print(f"[mainVirtualBkgd - INFO]: Grabbed background image.")
    print(f"[mainVirtualBkgd - INFO]: Please come back.")


  # Histogram equalise the bkgd image so that we can see it.
  hist_equ_bkgd_bgr_img = histEqu(bkgd_bgr_img)


  # Pause for 2 seconds for the user to come back into shot.
  time.sleep(2.0)


  # Start doing a stream...
  while True:
    # Capture an image
    bgr_img = captureImg(cap)

    # Remove the background
    bgr_detected_img = diffBkgd(bgr_img, bkgd_bgr_img)
    bgr_detected_img = resizeImage(bgr_detected_img)

    # Show images
    viewImage(frameName+' | bgr_detected_img', bgr_detected_img)

    c = cv2.waitKey(50)
    if c == 27: # esc key
        break


  # Show histogram equaled bkgd img
  viewImage(frameName+' | bkgd_bgr_img | hist_equ', hist_equ_bkgd_bgr_img)
  # Show image as persistant, so that we know that it has been captured.
  viewImagePersistant(frameName+' | bkgd_bgr_img', bkgd_bgr_img)

  camFinish(cap)
  closeAllImages()

  print("[mainVirtualBkgd - INFO]: Finished.")

#-------------------------------------------------------------------------------
def mainStraight():
  import sys, time
  frameName = f"Showing webcam"

  print("[mainVirtualBkgd - INFO]: Starting mainVirtualBkgd.py")
  print("[mainVirtualBkgd - HELP]: To finish the program, please close the images.")

  # See if we can find the image file given
  try:
    # sys.argv[0] is the script name
    imageFileName = sys.argv[1]

    # Read in the wanted image
    bkgd_bgr_img = readImage(imageFileName)
    print(f"[mainVirtualBkgd - DATA]: size(bkgd_bgr_img) = {bkgd_bgr_img.shape}")
  except:
    print(f"[mainVirtualBkgd - WARN]: No image specified...")
    print(f"[mainVirtualBkgd - INFO]: Grabbing background image from webcam...")
    print(f"[mainVirtualBkgd - INFO]: Please move away.")
    time.sleep(1.5)

    # Start up the camera
    cap = camStart()
    bkgd_bgr_img = captureImg(cap)
    print(f"[mainVirtualBkgd - INFO]: Grabbed background image.")
    print(f"[mainVirtualBkgd - INFO]: Please come back.")


  # Remove the background
  bgr_detected_img = diffBkgdOtsuBGR(bgr_img, bkgd_bgr_img)
  bgr_detected_img = resizeImage(bgr_detected_img)

  # Show image as persistant, so that we know that it has been captured.
  viewImagePersistant(frameName+' | bkgd_bgr_img', bkgd_bgr_img)

  camFinish(cap)
  closeAllImages()

  print("[mainVirtualBkgd - INFO]: Finished.")

#-------------------------------------------------------------------------------
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
  groups_img = removeBkgd(bgr_img)
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



