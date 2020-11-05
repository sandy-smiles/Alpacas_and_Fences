################################################################################

# Alpacas & Fences - viewCamera
# Authors: 470203101, 470386390, 470345850

# In order to run this file alone:
# $ python viewCamera.py

# This script aids in the collection of web camera images.

# Code modified from https://subscription.packtpub.com/book/application_development/9781785283932/3/ch03lvl1sec28/accessing-the-webcam

################################################################################
# Imports
################################################################################
import cv2

################################################################################
# Constants
################################################################################

################################################################################
# Functions
################################################################################
# camStart
# Returns the camera image capturing object.
# Input:
#   - N/A
# Output:
#   - cap -> camera image capturing object
def camStart():
  cap = cv2.VideoCapture(0)
  return cap

#-------------------------------------------------------------------------------
# captureImg
# Captures and image from the web camera using the given camera image capturing object
# Input:
#   - cap -> camera image capturing object
# Output:
#   - N/A
# Note:
#   - Captured image is shown on the screen.
def captureImg(cap):
  # Check if the webcam is opened correctly
  if not cap.isOpened():
    raise IOError("Cannot open webcam")

  ret, frame = cap.read()
  #frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
  cv2.imshow('Captured Image', frame)

#-------------------------------------------------------------------------------
# camFinish
# Releases the camera image capturing object and closes all open image showing windows.
# Input:
#   - cap -> camera image capturing object
# Output:
#   - N/A
def camFinish(cap):
  cap.release()
  cv2.destroyAllWindows()

################################################################################
# Main
################################################################################
if __name__ == "__main__":
  print("[viewCamera - INFO]: Starting viewCamera.py")
  print("[viewCamera - HELP]: To stop the webcam stream, please hit the esc key.")

  cap = camStart()

  while True:
    captureImg(cap)
    c = cv2.waitKey(1)
    if c == 27: # esc key
        break

  camFinish(cap)
  print("[viewCamera - INFO]: Finished viewCamera.py")


