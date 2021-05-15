'''
Created on 14-May-2021

@author: Avinash
'''

from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, Activation, ZeroPadding2D,Reshape, Permute
from keras.models import Sequential

import keras.backend as K
K.set_image_data_format('channels_first')

def VGG_FCN(width, height, depth, classes, weightsPath=None):
    
    data_shape = width*height
    pdsize = (1,1)    
    ksize = 3
    
    net = Sequential()
    net.add(ZeroPadding2D((1,1),input_shape=(depth,height,width)))
    net.add(Convolution2D(64, 3, 3, activation='relu'))
    net.add(ZeroPadding2D((1,1)))
    net.add(Convolution2D(64, 3, 3, activation='relu'))
    net.add(MaxPooling2D((2,2), strides=(2,2)))

    net.add(ZeroPadding2D((1,1)))
    net.add(Convolution2D(128, 3, 3, activation='relu'))
    net.add(ZeroPadding2D((1,1)))
    net.add(Convolution2D(128, 3, 3, activation='relu'))
    net.add(MaxPooling2D((2,2), strides=(2,2)))

    net.add(ZeroPadding2D((1,1)))
    net.add(Convolution2D(256, 3, 3, activation='relu'))
    net.add(ZeroPadding2D((1,1)))
    net.add(Convolution2D(256, 3, 3, activation='relu'))
    net.add(ZeroPadding2D((1,1)))
    net.add(Convolution2D(256, 3, 3, activation='relu'))
    net.add(ZeroPadding2D((1,1)))
    net.add(Convolution2D(256, 3, 3, activation='relu'))
    net.add(MaxPooling2D((2,2), strides=(2,2)))

    net.add(ZeroPadding2D((1,1)))
    net.add(Convolution2D(512, 3, 3, activation='relu'))
    net.add(ZeroPadding2D((1,1)))
    net.add(Convolution2D(512, 3, 3, activation='relu'))
    net.add(ZeroPadding2D((1,1)))
    net.add(Convolution2D(512, 3, 3, activation='relu'))
    net.add(ZeroPadding2D((1,1)))
    net.add(Convolution2D(512, 3, 3, activation='relu'))
    net.add(MaxPooling2D((2,2), strides=(2,2)))

    net.add(ZeroPadding2D((1,1)))    
    net.add(Convolution2D(512, 3, 3, activation='relu'))
    net.add(ZeroPadding2D((1,1)))        
    net.add(Convolution2D(512, 3, 3, activation='relu'))
    net.add(ZeroPadding2D((1,1)))        
    net.add(Convolution2D(512, 3, 3, activation='relu'))
    net.add(ZeroPadding2D((1,1)))    
    net.add(Convolution2D(512, 3, 3, activation='relu'))
    
    #===================================== Decoder Layers =============================================
    
    upsize = (2,2)
    # deConv Layer
    net.add(UpSampling2D(size=upsize))
    net.add(ZeroPadding2D(padding=pdsize))
    net.add(Convolution2D(64,ksize,ksize,border_mode='valid'))
            
    net.add(UpSampling2D(size=upsize))
    net.add(ZeroPadding2D(padding=pdsize))
    net.add(Convolution2D(64,ksize,ksize,border_mode='valid'))
                
    net.add(UpSampling2D(size=upsize))
    net.add(ZeroPadding2D(padding=pdsize))
    net.add(Convolution2D(32,ksize,ksize,border_mode='valid'))    

    net.add(UpSampling2D(size=upsize))
    net.add(ZeroPadding2D(padding=pdsize))
    net.add(Convolution2D(16,ksize,ksize,border_mode='valid'))
            
    #================================== Final Layer with softmax =======================================
    
    net.add(Convolution2D(classes,1,1,border_mode='valid'))    
    net.add(Reshape((classes,data_shape),input_shape=(classes,height,width)))
    net.add(Permute((2,1)))
    net.add(Activation("softmax"))
        
    #=========================== If a pre trained model is supplied =====================================
    if weightsPath is not None:
        net.load_weights(weightsPath)
           
    return net

