import os
import keras
import numpy as np
import pandas as pd
from scipy.misc import imread

from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape, InputLayer
from keras.regularizers import L1L2

import tensorflow

seed=128
rng=np.random.RandomState(seed)

root_dir=os.path.abspath('.')
data_dir = os.path.join(root_dir,'Data')
"""
import mnist

train_images = mnist.train_images()
train_labels = mnist.train_labels()

test_images = mnist.test_images()
test_labels = mnist.test_labels()"""


train=pd.read_csv("/home/petrichor/Downloads/mnist/train.csv")
print "loaded"
#print len(mnist.images)

array=np.reshape(train.iloc[1,1:].values,(28,28))
print array.shape	
"""
img = imread(train.iloc[1,1:].values, flatten=True)

pylab.imshow(img, cmap='gray')
pylab.axis('off')
pylab.show()"""

pixels = np.array(train.iloc[1,1:].values, dtype='uint8')
# Reshape the array into 28 x 28 array (2-dimensional array)
pixels = pixels.reshape((28, 28))
import cv2
cv2.imshow("aaa", pixels)
cv2.waitKey(0)
cv2.destroyAllWindows()
img=np.array(array,np.uint8)
cv2.imshow("ii",img)
cv2.waitKey(0)
