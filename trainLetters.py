################################################################################

# Alpacas & Fences - viewCamera
# Authors: 470203101, 470386390, 470354850

# In order to run this file alone:
# $ python viewCamera.py

# This script aids in the collection of web camera images.

# Code modified from https://subscription.packtpub.com/book/application_development/9781785283932/3/ch03lvl1sec28/accessing-the-webcam

################################################################################
# Imports
################################################################################
import cv2
import numpy as np
import matplotlib.pyplot as plt
import idx2numpy

from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split

from skimage.feature import hog

################################################################################
# Constants
################################################################################

################################################################################
# Functions
################################################################################

################################################################################
# Main
################################################################################
if __name__ == "__main__":
  
  image_size = 28 # width and length
  no_of_different_labels = 26 #  i.e. 0, 1, 2, 3, ..., 9
  image_pixels = image_size * image_size

  # Training data
  all_train_X = idx2numpy.convert_from_file('c:/Users/strat/OneDrive/Desktop/majorProjectCV/emnist-letters-train-images-idx3-ubyte')
  all_train_y = idx2numpy.convert_from_file('c:/Users/strat/OneDrive/Desktop/majorProjectCV/emnist-letters-train-labels-idx1-ubyte')
  im_Tracker = 0
  plot_Number = 1

  new_train_X = []
  new_train_y = []

  n_Samples = len(all_train_y)

  # Starting off with letters R O Y G C B P M
  letters_To_Track = [18, 15, 25, 7, 3, 2, 16, 13]

  print('Train: X=%s, y=%s' % (all_train_X.shape, all_train_y.shape))

  # Looping through all training data
  for i in range(n_Samples):
   
    # Current image to train
    current_Im = all_train_X[i].transpose()    

    if all_train_y[i] in letters_To_Track:

      # Normalising image
      processed_Im = current_Im / 255
      #print(processed_Im)

      hog_Feature = hog(processed_Im, orientations=8, pixels_per_cell=(4, 4), 
                    cells_per_block=(1, 1), visualize=False, multichannel=False)

      # Adding onto the feature vector
      new_train_X.append(hog_Feature)
      new_train_y.append(all_train_y[i])

      #im_Tracker += 1

      # define subplot
      #plt.subplot(5, 5, plot_Number)   

      # plot raw pixel data
      #plt.imshow(hog_image, cmap=plt.get_cmap('gray'))
      #plot_Number += 1

      #if plot_Number > 10000 or :
      #  break

  # show the figure
  #plt.show()

  #print(new_train_y)

  #for i in fd:
  #  print(i)

  # Implementing a SVM with a Linear Kernel
  X_train, X_test, y_train, y_test = train_test_split(new_train_X, new_train_y, test_size=0.1) # 90% training and 10% test
  clf = svm.SVC(kernel='linear')
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)

  # Assessing metrics
  print(y_test)
  print(y_pred)
  print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

  print("END PROGRAM")

  #while True:
    


