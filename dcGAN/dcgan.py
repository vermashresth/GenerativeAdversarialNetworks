import keras
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape, InputLayer, Conv2D, Dropout, Activation,BatchNormalzation, UpSampling2D, Conv2DTranspose

from keras.layers.advanced_activations import LeakyReLU 
from keras.optimisers import RMSprop


class adv():
    def __init__(self,img_rows=28,img_cols=28,channel=1):
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
        
        return self.D
        
        
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
        
        self.G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        
        self.G.add(Conv2DTranspose(1, 5, padding='same'))
        self.G.add(Activation('sigmoid'))
        self.G.summary()
        
        return self.G
        
    def dis_model(self):
        optimiser=RMSprop(lr=0.0008,clipvalue=1.0,decay=6e-8)
        self.DM=Sequential()
        self.DM.add(self.createdis())
        self.DM.compile(loss='binary_crossentropy',optimiser=optimiser,metrics=['accuracy'])
        
    def adv_model(self):
        optimizer = RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8)
        self.AM = Sequential()
        self.AM.add(self.creategen())
        self.AM.add(self.createdis())
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])

train=pd.read_csv("/home/petrichor/Downloads/mnist/train.csv")


temp=[]
for i in range(1,train.shape[0]):
	temp.append(np.reshape(train.iloc[i,1:].values,(28,28,1)))
print len(temp)
train_x=np.stack(temp)       
        
class MNIST():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channel = 1
        self.x_train=train_x
        self.adv=adv()
        self.discriminator=self.adv.dis_model()
        self.adversarial=self.adv.adv_model()
        self.generator=self.adv.creategen()
        
    def train(self, train_steps=2000, batch_size=256, save_interval=0): 
        noise_input = None
            if save_interval>0:
                noise_input=np.random.uniform(-1.0,1.0,size=[16,100])
            for i in range(train_steps):
                images_train=x_train[np.random.randint(0,self.x_train.shape[0],size=batch_size),:,:,:]
                noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
                images_fake=self.generator.predict(noise)
                x=np.concatenate((images_train,images_fake))
                y=np.ones([batch_sie*2,1])
                y[batch_size:,:]=0
                d_loss=self.discriminator.train_on_batch(x,y)
                
                y=np.ones([batch_size,1])
                noise=np.random.uniform(-1.0,1.0,size=[batch_size,100])
                a_loss=self.adversarial.train_on_batches(noise,y)
                log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
                log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
                print(log_mesg)
                if save_interval>0:
                    if (i+1)%save_interval==0:
                        self.plot_images(save2file=True, samples=noise_input.shape[0],\
                            noise=noise_input, step=(i+1))

    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = 'mnist.png'
        if fake:
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
            else:
                filename = "mnist_%d.png" % step
            images = self.generator.predict(noise)
        else:
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]

        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()
            
if __name__=='__main__':
    mnist=MNIST()
    mnist.train(train_steps=10000, batch_size=256, save_interval=500)
    mnist.plot_images(fake=True)
    mnist.plot_images(fake=False, save2file=True)
                
        
        
        
        
        
        
        
