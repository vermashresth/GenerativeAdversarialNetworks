import keras
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape, InputLayer, Conv2D, Dropout, Activation,BatchNormalzation, UpSampling2D, Conv2DTranspose

from keras.layers.advanced_activations import LeakyReLU 

class adv():
    def __init__(self,img_rows,img_cols,channel):
        self.img_rows = img_rows
        self.img_cols=img_cols
        self.channel=channel
        self.D=None
        self.G=None
        
    def createdis(self,):
        self.D = Sequential()
        depth = 64
        dropout = 0.4
        input_shape = (self.img_rows, self.img_cols, self.channel)
        self.D.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape, padding='same', activation=LeakyReLU(alpha=0.2)))
        self.D.add(Dropout(dropout))
        self.D.add(Conv2D(depth*2, 5, strides = 2, padding='same', activation =LeakyReLU(alpha=0.2)))
        self.D.add(Dropout(dropout))
        self.D.add(Conv2D(depth*4, 5, strides =2, padding ='same', activation=LeakyReLU(alpha=0.2)))
        self.D.add(Dropout(dropout))
        self.D.add(Conv2D(depth*8, 5, strides =1, padding='same', activation=LeakyReLU(alpha=0.2)))
        self.D.add(Dropout(dropout))
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        self.D.summary()
        
    def creategen(self,):
        self.G = Sequential()
        dropout = 0.4
        depth =64+64+64+64
        dim =7
	  self.G.add(Dense(dim*dim*depth,input_dim=100))
        self.G.add(BatchNormalzation(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(Reshape((dim, dim, depth)))
        self.G.add(Dropout(dropout))
        
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/2),5,padding='same'))
        self.G.add(BatchNormalzation(momentum=0.9))
        
        self.G.add(Activation('relu'))
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        
        
        
        
        
        
        
