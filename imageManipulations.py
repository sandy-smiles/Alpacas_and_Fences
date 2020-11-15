################################################################################

# Alpacas & Fences - imageManipulations
# Authors: 470203101, 470386390, 470345850

# In order to run this file alone:
# $ python imageManipulations.py

# This script aids in the manipulations of images needed for other scripts.

################################################################################
# Imports
################################################################################
from skimage.measure import compare_ssim
import imutils
import cv2
import numpy as np
from PIL import Image

################################################################################
# Constants
################################################################################
hist_equ_tile_size = 8

################################################################################
# Functions
################################################################################
# histEqu
# Removes the background of the given rgb_img through diffing the bkgd_img
# Input:
#   - bgr_img | cv2 image -> image object/data
#   - bkgd_img | cv2 image -> image object/data
# Output:
#   - N/A
# Notes:
#   - Implementation guided from https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html
def histEqu(bgr_img):
  # split the image into its separate channels
  b_img, g_img, r_img = cv2.split(bgr_img)

  # CLAHE application object
  clahe = cv2.createCLAHE(clipLimit = 2.0,
    tileGridSize = (hist_equ_tile_size, hist_equ_tile_size))

  # red channel application
  r_clahe_img = clahe.apply(r_img)
  # green channel application
  g_clahe_img = clahe.apply(g_img)
  # blue channel application
  b_clahe_img = clahe.apply(b_img)

  return cv2.merge((b_clahe_img, g_clahe_img, r_clahe_img))

#-------------------------------------------------------------------------------
# diffImgs
# Removes the background of the given rgb_img through diffing the bkgd_img
# Input:
#   - bgr_img | cv2 image -> image object/data
#   - bkgd_img | cv2 image -> image object/data
# Output:
#   - N/A
def diffImgs(gray_img, bkgd_gray_img):
  (score, diff) = compare_ssim(gray_img, bkgd_gray_img, full=True)
  diff = (diff * 255).astype("uint8")
  print(f"[diffImgs - INFO]: SSIM = {score}")

  _, thresh = cv2.threshold(diff, 0, 255,
    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

  return thresh

#-------------------------------------------------------------------------------
# normImg
# Normalise image
# Input:
#   - img | cv2 image -> image object/data
# Output:
#   - img | cv2 image -> image object/data with values between 0.0 and 1.0
def normImg(img):
  img = np.subtract(img, np.amin(img))
  img = np.divide(img, np.amax(img))
  return img

#-------------------------------------------------------------------------------
# resizeImage
# Resize the image.
# Input:
#   - image | cv2 image -> image object/data
#   - scale | 0.0 -> 1.0 | float -> scale
# Output:
#   - image | cv2 image -> image object/data
# Note:
#   - Image will fit screen.
def resizeImage(image, scale=0.5):
  # Resize the image
  img_h, img_w = image.shape[0:2]
  new_w = int(img_w * scale)
  new_h = int(img_h * scale)
  # dim = (width, height)
  dim = (new_w, new_h)

  # resize image
  return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

################################################################################
# Main
################################################################################
def main():
  pass

if __name__ == "__main__":
  main()


