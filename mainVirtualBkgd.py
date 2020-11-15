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
from imageBkgds import *

################################################################################
# Constants
################################################################################


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
  return bgr_img

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

if __name__ == "__main__":
  main()



