################################################################################

# Alpacas & Fences - imageBkgds
# Authors: 470203101, 470386390, 470345850

# In order to run this file alone:
# $ python imageBkgds.py

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
# diffBkgdThreshBGR
# Removes the background of the given rgb_img through diffing the bkgd_img
# Input:
#   - bgr_img | cv2 image -> image object/data
#   - bkgd_img | cv2 image -> image object/data
# Output:
#   - N/A
def diffBkgdThreshBGR(bgr_img, bkgd_img):
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
# diffBkgdBGRThresh
# Removes the background of the given rgb_img through diffing the bkgd_img
# Input:
#   - bgr_img | cv2 image -> image object/data
#   - bkgd_img | cv2 image -> image object/data
# Output:
#   - N/A
def diffBkgdBGRThresh(bgr_img, bkgd_img):
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
# diffBkgdOtsuBGR
# Removes the background of the given rgb_img.
# Input:
#   - bgr_img | cv2 image -> image object/data
# Output:
#   - N/A
def diffBkgdOtsuBGR(bgr_img):
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
# diffBkgdOtsuGray
# Removes the background of the given rgb_img.
# Input:
#   - bgr_img | cv2 image -> image object/data
# Output:
#   - N/A
def diffBkgdOtsuGray(bgr_img):
  gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
  ret_gray, thresh_gray = cv2.threshold(gray_img, 0,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
  print(f"[removeBkgd - INFO]: ret_gray = {ret_gray}")

  return thresh_gray

#-------------------------------------------------------------------------------
# diffBkgdModels
# Removes the background of the given rgb_img.
# Input:
#   - bgr_img | cv2 image -> image object/data
# Output:
#   - N/A
def diffBkgdModels(bgr_img, model='fcn'):
  if model == 'fcn':
    model = models.segmentation.fcn_resnet101(pretrained=True).eval()
  else:
    model = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()

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
  out = model(inp)['out']
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

