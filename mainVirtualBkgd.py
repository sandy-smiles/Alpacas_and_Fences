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

################################################################################
# Constants
################################################################################


################################################################################
# Functions
################################################################################
# diffBkgd
# Removes the background of the given rgb_img through diffing the bkgd_img
# Input:
#   - bgr_img | cv2 image -> image object/data
#   - bkgd_img | cv2 image -> image object/data
# Output:
#   - N/A
def diffBkgd2(bgr_img, bkgd_img):
  # split the image into its separate channels
  b_img, g_img, r_img = cv2.split(bgr_img)
  b_bkgd_img, g_bkgd_img, r_bkgd_img = cv2.split(bkgd_img)

  # red channel application
  r_thresh =  diffImgs(r_img, r_bkgd_img)
  # green channel application
  g_thresh =  diffImgs(g_img, g_bkgd_img)
  # blue channel application
  b_thresh =  diffImgs(b_img, b_bkgd_img)

  t_thresh = r_thresh & g_thresh & b_thresh

  # red channel application
  r_diff_img = cv2.bitwise_and(r_img, r_img, mask=t_thresh)
  # green channel application
  g_diff_img = cv2.bitwise_and(g_img, g_img, mask=t_thresh)
  # blue channel application
  b_diff_img = cv2.bitwise_and(b_img, b_img, mask=t_thresh)

  diff_img = cv2.merge((b_diff_img, g_diff_img, r_diff_img))

  return diff_img

#-------------------------------------------------------------------------------
# diffBkgd
# Removes the background of the given rgb_img through diffing the bkgd_img
# Input:
#   - bgr_img | cv2 image -> image object/data
#   - bkgd_img | cv2 image -> image object/data
# Output:
#   - N/A
def diffBkgd(bgr_img, bkgd_img):
  # split the image into its separate channels
  b_img, g_img, r_img = cv2.split(bgr_img)
  b_bkgd_img, g_bkgd_img, r_bkgd_img = cv2.split(bkgd_img)

  # red channel application
  r_thresh =  diffImgs(r_img, r_bkgd_img)
  r_diff_img = cv2.bitwise_and(r_img, r_img, mask=r_thresh)
  # green channel application
  g_thresh =  diffImgs(g_img, g_bkgd_img)
  g_diff_img = cv2.bitwise_and(g_img, g_img, mask=g_thresh)
  # blue channel application
  b_thresh =  diffImgs(b_img, b_bkgd_img)
  b_diff_img = cv2.bitwise_and(b_img, b_img, mask=b_thresh)

  diff_img = cv2.merge((b_diff_img, g_diff_img, r_diff_img))

  return diff_img

#-------------------------------------------------------------------------------
# removeBkgd Deprecated
# Removes the background of the given rgb_img.
# Input:
#   - bgr_img | cv2 image -> image object/data
# Output:
#   - N/A
def removeBkgdDep(bgr_img):
  blue_img, green_img, red_img = cv2.split(bgr_img)

  ret_red, thresh_red = cv2.threshold(red_img, 0,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
  print(f"[removeBkgd - INFO]: size(thresh_red) = {thresh_red.shape}")
  print(f"[removeBkgd - INFO]: ret_red = {ret_red}")


  ret_green, thresh_green = cv2.threshold(green_img, 0,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
  print(f"[removeBkgd - INFO]: size(thresh_green) = {thresh_green.shape}")
  print(f"[removeBkgd - INFO]: ret_green = {ret_green}")


  ret_blue, thresh_blue = cv2.threshold(blue_img, 0,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
  print(f"[removeBkgd - INFO]: size(thresh_blue) = {thresh_blue.shape}")
  print(f"[removeBkgd - INFO]: ret_blue = {ret_blue}")

  ret = int(ret_red+ret_green+ret_blue)/3
  thresh = (thresh_red+thresh_green+thresh_blue)
  thresh = np.subtract(thresh, np.amin(thresh))
  thresh = np.divide(thresh, np.amax(thresh))

  return thresh

#-------------------------------------------------------------------------------
# removeBkgd Deprecated
# Removes the background of the given rgb_img.
# Input:
#   - bgr_img | cv2 image -> image object/data
# Output:
#   - N/A
def removeBkgdDepDep(bgr_img):
  gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
  ret_gray, thresh_gray = cv2.threshold(gray_img, 0,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
  print(f"[removeBkgd - INFO]: ret_gray = {ret_gray}")

  return thresh_gray

#-------------------------------------------------------------------------------
# removeBkgd
# Removes the background of the given rgb_img.
# Input:
#   - bgr_img | cv2 image -> image object/data
# Output:
#   - N/A
def removeBkgd(bgr_img):
  fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()
  dpl = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()

  trf = T.Compose([T.Resize(256),
                   T.CenterCrop(224),
                   T.ToTensor(), 
                   T.Normalize(mean = [0.485, 0.456, 0.406], 
                               std = [0.229, 0.224, 0.225])])


  # Change the given CV image into PIL
  rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
  rgb_img_pil = Image.fromarray(rgb_img)


  # Apply transforms to the image
  inp = trf(rgb_img_pil).unsqueeze(0)

  # Pass the input through the net
  out = fcn(inp)['out']
  print(f"[removeBkgd - DATA]: out.shape = {out.shape}")

  # Find out which types of objects there were
  om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
  print(f"[removeBkgd - DATA]: om.shape = {om.shape}")
  print(f"[removeBkgd - DATA]: np.unique(om) = {np.unique(om)}")

  # Decode the different objects into colours
  label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

  r = np.zeros_like(om).astype(np.uint8)
  g = np.zeros_like(om).astype(np.uint8)
  b = np.zeros_like(om).astype(np.uint8)

  nc = [0,15];
  for l in nc:
    idx = om == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]
    
  bgr = np.stack([b, g, r], axis=2)
  return bgr

#-------------------------------------------------------------------------------
# replaceBkgd
# Replaces the background of the given rgb_img with the bkgd_img
# Input:
#   - bgr_img | cv2 image -> image object/data
#   - bkgd_img | cv2 image -> image object/data
# Output:
#   - N/A
def replaceBkgd(bgr_img, bkgd_img):
  pass

################################################################################
# Main
################################################################################
def mainDep():
  import sys

  import torch
  from torchvision import models
  import torchvision.transforms as T

  print("[mainVirtualBkgd - INFO]: Starting mainVirtualBkgd.py")
  print("[mainVirtualBkgd - HELP]: To finish the program, please close the image window.")

  # See if we can find the image file given
  try:
    # sys.argv[0] is the script name
    imageFileName = sys.argv[1]
  except:
    print(f"[mainVirtualBkgd - ERROR]: You forgot to specify all needed arguments.")
    print(f"[mainVirtualBkgd - ERROR]: Try running:\n\tpython3 mainVirtualBkgd.py ./virtualBkgdImgs/IMG20190812150924.jpg")
    return

  # Read in the wanted image
  bgr_img = readImage(imageFileName)
  print(f"[mainVirtualBkgd - DATA]: size(bgr_img) = {bgr_img.shape}")

  # Remove the background
  bgr_detected_img = removeBkgd(bgr_img)
  print(f"[mainVirtualBkgd - DATA]: size(bgr_detected_img) = {bgr_detected_img.shape}")

  # Show images
  frameName = f"Showing image {imageFileName}"
  bgr_detected_img = resizeImage(bgr_detected_img)
  viewImagePersistant(frameName+' | bgr_detected_img', bgr_detected_img)

  closeAllImages()

  print("[mainVirtualBkgd - INFO]: Finished.")


#-------------------------------------------------------------------------------
def main2():
  import sys

  print("[mainVirtualBkgd - INFO]: Starting mainVirtualBkgd.py")
  print("[mainVirtualBkgd - HELP]: To finish the program, please hit esc.")

  # Start up the camera
  cap = camStart()

  # Start doing a stream...
  while True:
    # Capture an image
    bgr_img = captureImg(cap)

    # Remove the background
    bgr_detected_img = removeBkgdDepDep(bgr_img)

    # Show images
    frameName = f"Showing webcam"
    bgr_detected_img = resizeImage(bgr_detected_img)
    viewImage(frameName+' | bgr_detected_img', bgr_detected_img)

    c = cv2.waitKey(1)
    if c == 27: # esc key
        break


  closeAllImages()

  print("[mainVirtualBkgd - INFO]: Finished.")


#-------------------------------------------------------------------------------
def main():
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

    # Show images
    bgr_detected_img = resizeImage(bgr_detected_img)
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

if __name__ == "__main__":
  main()



