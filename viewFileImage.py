################################################################################

# Alpacas & Fences - viewFileImage
# Authors: 470203101, 470386390, 470345850

# In order to run this file alone:
# $ python viewFileImage.py ./virtualBkgdImgs/IMG20190812150924.jpg

# This script aids in the collection of an image from file.

################################################################################
# Imports
################################################################################
import sys
import cv2

################################################################################
# Constants
################################################################################

################################################################################
# Functions
################################################################################
# readImage
# Returns the read in image file.
# Input:
#   - imageFileName | str -> image file name
# Output:
#   - image
def readImage(imageFileName):
  return  cv2.imread(imageFileName)

#-------------------------------------------------------------------------------
# resize2ScreenImage
# Resize the image to better fit on the screen.
# Input:
#   - image | cv2 image -> image object/data
# Output:
#   - image | cv2 image -> image object/data
# Note:
#   - Image will fit screen.
def resize2ScreenImage(image):
  # Resize the image
  screen_w = 800;
  screen_h = 600;
  img_h = len(image[0])
  img_w = len(image)
  scale_const = 0.75
  #scale_val = scale_const
  scale_val = {
    True: screen_w/img_w*scale_const, 
    False: screen_h/img_h*scale_const,
  }[img_w > img_h] # percent of original size
  width = int(img_h * scale_val)
  height = int(img_w * scale_val)
  dim = (width, height)

  # resize image
  return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

#-------------------------------------------------------------------------------
# viewImagePersistant
# Show the given image in a frame with specified name.
# Input:
#   - frameName | str -> frame name
#   - image | cv2 image -> image object/data
# Output:
#   - N/A
# Note:
#   - Image is shown on the screen.
#   - Code will not progress from this point, until the image has been closed.
def viewImagePersistant(frameName, image):
  cv2.imshow(frameName, image)

  # Wait until the window has been closed
  while cv2.getWindowProperty(frameName, cv2.WND_PROP_VISIBLE) > 0:
    keyCode = cv2.waitKey(50)

#-------------------------------------------------------------------------------
# viewImage
# Show the given image in a frame with specified name.
# Input:
#   - frameName | str -> frame name
#   - image | cv2 image -> image object/data
# Output:
#   - N/A
# Note:
#   - Image is shown on the screen.
#   - Code will not progress from this point, until the image has been closed.
def viewImage(frameName, image):
  cv2.imshow(frameName, image)

#-------------------------------------------------------------------------------
# closeAllImages
# Closes all open image frames.
# Input:
#   - N/A
# Output:
#   - N/A
def closeAllImages():
  cv2.destroyAllWindows()


################################################################################
# Main
################################################################################
def main():
  print("[viewFileImage - INFO]: Starting viewImage.py")
  print("[viewFileImage - HELP]: To finish the program, please close the image window.")

  # See if we can find the image file given
  try:
    # sys.argv[0] is the script name
    imageFileName = sys.argv[1]
  except:
    print(f"[viewFileImage - ERROR]: You forgot to specify all needed arguments.")
    print(f"[viewFileImage - ERROR]: Try running:\n\tpython3 viewFileImage.py ./virtualBkgdImgs/IMG20190812150924.jpg")
    return

  rgb_img = readImage(imageFileName)
  frameName = f"Showing image {imageFileName}"
  viewImage(frameName+'0', rgb_img)
  viewImagePersistant(frameName+'1', rgb_img)

  closeAllImages()

  print("[viewFileImage - INFO]: Finished.")

if __name__ == "__main__":
  main()


