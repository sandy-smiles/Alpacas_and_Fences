################################################################################

# Alpacas & Fences - imageClusterings.py
# Authors: 470203101, 470386390, 470345850

# In order to run this file alone:
# $ python imageClusterings.py ./virtualBkgdImgs/IMG20190812150924.jpg

################################################################################
# Imports
################################################################################
import cv2
import numpy as np
import math
from PIL import Image

################################################################################
# Constants
################################################################################
COLOR_MAX = 255
COLOR_MIN = 0

C_THRESH = 20
D_TILE = 1

################################################################################
# Functions
################################################################################
# removeListDup
# Removes duplicates from a list.
# Input:
#   - l | list
# Output:
#   - l | list
# Note:
#   - (0, 1) and (-0, 1) are treated as being the same thing.
def removeListDup(l):
  return list(dict.fromkeys(l))

#-------------------------------------------------------------------------------
# createCheckList
# Creates the list of tiles to look at
# relative to the currently looked at tile.
# Input:
#   - r | int -> range around current tile.
# Output:
#   - list of tuples
def createCheckList(r):
  l = []
  for i in range(1, r+1):
    for j in range(i):
      l.append((i,j))
      l.append((-i,j))
      l.append((-i,-j))
      l.append((i,-j))
      l.append((j,i))
      l.append((-j,i))
      l.append((-j,-i))
      l.append((j,-i))
    l.append((i,i))
    l.append((-i,i))
    l.append((-i,-i))
    l.append((i,-i))
  return removeListDup(l)

#-------------------------------------------------------------------------------
# createGroupsBGR
# Returns pixel groups.
# Input:
#   - bgr_img | cv2 image -> image object/data
#   - c_thresh | 0 - 255 int -> colour threshold
#   - r_tile | int -> # top left corner tiles to look at.
# Output:
#   - gray_img | cv2 image -> image object/data
def createGroupsBGR(bgr_img, c_thresh=C_THRESH, d_tile=D_TILE):
  b_img, g_img, r_img = cv2.split(bgr_img)
  # For each pixel, go down
  row, col, channels = bgr_img.shape
  gp_img = np.zeros((row, col))-1.0;
  gp_dict = {}

  tiles = createCheckList(d_tile)

  def peekHue(r, c):
    return (b_img[r][c], g_img[r][c], r_img[r][c])

  def peekGroup(r, c):
    return gp_img[r][c]

  def peekSurround(r, c, tiles):
    #print(f"[peekSurround - DATA]: r = {r}, c = {c}")
    gps = {}
    # obtain current tile's colour (hue)
    p_b, p_g, p_r = peekHue(r, c)
    for d_r, d_c in tiles:
      tile_r = r+d_r
      tile_c = c+d_c
      # check tile is possible
      if tile_r < 0 or tile_c < 0 or tile_r >= row or tile_c >= col:
        # no current group
        # therefore can not compare
        continue
      # look at tile group
      g = peekGroup(tile_r, tile_c)
      if g == -1.0:
        # no current group
        # therefore can not compare
        continue
      # look at tile colour (hue)
      h_b, h_g, h_r = peekHue(tile_r, tile_c)
      h_b_diff = min(abs(p_b-h_b), abs(p_b+COLOR_MAX-h_b))
      h_g_diff = min(abs(p_g-h_g), abs(p_g+COLOR_MAX-h_g))
      h_r_diff = min(abs(p_r-h_r), abs(p_r+COLOR_MAX-h_r))
      #h_diff = math.sqrt(h_b_diff**2 + h_g_diff**2 + h_r_diff**2)
      # save the group num as the tile is within color hue thres
      if (0 <= h_b_diff) and (h_b_diff <= c_thresh) and (0 <= h_g_diff) and (h_g_diff <= c_thresh) and (0 <= h_g_diff) and (h_g_diff <= c_thresh):
      #if (0 <= h_diff) and (h_diff <= c_thresh):
        try:
          gps[g] += 1
        except:
          gps[g] = 1
    return gps

  # find groups
  gp_n = 0; # last group number
  for r in range(row):
    col_range = range(col)
    if r%2 != 0:
      col_range = reversed(col_range)
    print(f"[createGroups - DATA]: r = {r}, gp_n = {gp_n}")
    for c in col_range:
      gps = peekSurround(r, c, tiles)
      if len(gps.keys()) == 0:
        gp_img[r][c] = gp_n
        gp_n += 1
      else:
        k_max, v_max = -1, -1
        for k, v in gps.items():
          if v_max <= v:
            v_max = v
            k_max = k
        gp_img[r][c] = k_max

  return gp_img


#-------------------------------------------------------------------------------
# createGroupsGray
# Returns pixel groups.
# Input:
#   - gray_img | cv2 image -> image object/data
#   - c_thresh | 0 - 255 int -> colour threshold
#   - r_tile | int -> # top left corner tiles to look at.
# Output:
#   - gray_img | cv2 image -> image object/data
def createGroupsGray(gray_img, c_thresh=C_THRESH, d_tile=D_TILE):
  # For each pixel, go down
  row, col = gray_img.shape[0:2]
  gp_img = np.zeros((row, col), np.float32)-1.0;
  gp_dict = {}

  tiles = createCheckList(d_tile)

  def peekHue(r, c):
    return gray_img[r][c]

  def peekGroup(r, c):
    return gp_img[r][c]

  def peekSurround(r, c, tiles):
    #print(f"[peekSurround - DATA]: r = {r}, c = {c}")
    gps = {}
    # obtain current tile's colour (hue)
    p = peekHue(r, c)
    for d_r, d_c in tiles:
      tile_r = r+d_r
      tile_c = c+d_c
      # check tile is possible
      if tile_r < 0 or tile_c < 0 or tile_r >= row or tile_c >= col:
        # no current group
        # therefore can not compare
        continue
      # look at tile group
      g = peekGroup(tile_r, tile_c)
      if g == -1.0:
        # no current group
        # therefore can not compare
        continue
      # look at tile colour (hue)
      h = peekHue(tile_r, tile_c)
      h_diff = min(abs(p-h), abs(p+COLOR_MAX-h))
      # save the group num as the tile is within color hue thres
      if (0 <= h_diff) and (h_diff <= c_thresh):
        try:
          gps[g] += 1
        except:
          gps[g] = 1
    return gps

  # find groups
  gp_n = 0; # last group number
  for r in range(row):
    col_range = range(col)
    if r%2 != 0:
      col_range = reversed(col_range)
    print(f"[createGroups - DATA]: r = {r}, gp_n = {gp_n}")
    for c in col_range:
      gps = peekSurround(r, c, tiles)
      if len(gps.keys()) == 0:
        gp_img[r][c] = gp_n
        gp_n += 1
      else:
        k_max, v_max = -1, -1
        for k, v in gps.items():
          if v_max <= v:
            v_max = v
            k_max = k
        gp_img[r][c] = k_max

  return gp_img

################################################################################
# Main
################################################################################
def main():
  print("[imageClusterings - INFO]: Starting imageClusterings.py")
  import sys, time

  frameName = f"Showing webcam"

  print("[imageClusterings - HELP]: To finish the program, please close the images.")

  # See if we can find the image file given
  try:
    # sys.argv[0] is the script name
    imageFileName = sys.argv[1]
    print(f"[imageClusterings - INFO]: image specified as {imageFileName}")

    # Read in the wanted image
    bgr_img = readImage(imageFileName)
    print(f"[imageClusterings - DATA]: size(bgr_img) = {bgr_img.shape}")
    frameName = f"Select img"
  except:
    print(f"[imageClusterings - WARN]: No image specified...")
    print(f"[imageClusterings - INFO]: Grabbing background image from webcam...")
    print(f"[imageClusterings - INFO]: Please move away.")
    time.sleep(1.5)

    # Start up the camera
    cap = camStart()
    bgr_img = captureImg(cap)
    print(f"[imageClusterings - INFO]: Grabbed image.")
    camFinish(cap)
    print(f"[imageClusterings - INFO]: Please come back.")

  bgr_img_resized = resize2ScreenImage(bgr_img)
  viewImage(frameName+' | bgr_img', bgr_img_resized)
  gray_img = cv2.split(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY))[0]
  gray_img_resized = resize2ScreenImage(gray_img)
  viewImage(frameName+' | gray_img', gray_img_resized)

  # Imaging variables
  c_thresh = 200
  d_tile = 2;
  scale = 0.025

  # Remove the background
  scale = 0.025
  groups_img = createGroupsBGR(resizeImage(bgr_img, scale), c_thresh, d_tile)
  print(f"[imageClusterings - DATA]: groups_img.shape = {groups_img.shape}")
  print(f"[imageClusterings - DATA]: np.amax(groups_img) = {np.amax(groups_img)}")
  print(f"[imageClusterings - DATA]: np.amin(groups_img) = {np.amin(groups_img)}")
  groups_img = normImg(groups_img)
  groups_img_resized = resize2ScreenImage(groups_img)
  # Show image as persistant, so that we know that it has been captured.
  viewImage(f"{frameName} | groups_img | scale={scale}, c_thresh={c_thresh}, d_tile={d_tile}", groups_img_resized)

  # Remove the background
  scale = 0.05
  groups_img = createGroupsBGR(resizeImage(bgr_img, scale), c_thresh, d_tile)
  print(f"[imageClusterings - DATA]: groups_img.shape = {groups_img.shape}")
  print(f"[imageClusterings - DATA]: np.amax(groups_img) = {np.amax(groups_img)}")
  print(f"[imageClusterings - DATA]: np.amin(groups_img) = {np.amin(groups_img)}")
  groups_img = normImg(groups_img)
  groups_img_resized = resize2ScreenImage(groups_img)
  # Show image as persistant, so that we know that it has been captured.
  viewImagePersistant(f"{frameName} | groups_img | scale={scale}, c_thresh={c_thresh}, d_tile={d_tile}", groups_img_resized)

  closeAllImages()

  print("[imageClusterings - INFO]: Finished.")

if __name__ == "__main__":
  from viewFileImage import readImage, resize2ScreenImage, viewImage, viewImagePersistant, closeAllImages
  from viewCamera import camStart, captureImg, camFinish
  from imageManipulations import normImg, resizeImage

  print("[MAIN - INFO]: Starting...")
  main()
  print("[MAIN - INFO]: Finished.")


