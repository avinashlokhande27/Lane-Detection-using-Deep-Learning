'''
Created on 14-May-2021

@author: Avinash
'''

import os
import glob
import cv2
import NetPixel
import numpy as np
from lane_pre import crop_tiling
import keras.backend as K
K.set_image_data_format('channels_first')

ip_path='/data/binary_lane_bdd/Images/val/val_data_all/'

modelPath = '/data/binary_lane_bdd/Models/VGG_FCN3.h5'
data_format='.jpg'

imheight = 544
imwidth = 640
imdepth = 3
data_shape = imheight*imwidth
classes = 2

model = NetPixel.VGG_FCN(imwidth, imheight,imdepth,classes,modelPath)
model.summary()
              
Path2write=ip_path+'/output_vgg/'
os.makedirs(Path2write,exist_ok=True)
        
imageDir = sorted(glob.glob(ip_path+"/*"+data_format))

for tile in imageDir: 

    tileName = os.path.split(tile)[-1]
    tileNo = tileName.split(".j",1)
    im = cv2.imread(tile)
    hig,wid,dep = im.shape
    op_img= np.zeros((hig,wid),dtype=np.uint8)
    im1,im2,im3 =crop_tiling(im) 
    
    im1 = np.array(im1/255, dtype=np.float32)
    im2 = np.array(im2/255, dtype=np.float32)
              
    data = []
    data.append(np.rollaxis((im1),2))
    data.append(np.rollaxis((im2),2))
    temp = np.array(data)

    prob = model.predict(temp,verbose=2)
    
    
    prediction1 = np.argmax(prob[0],axis=-1)
    prediction2 = np.argmax(prob[1],axis=-1)
           
    prediction1 = np.reshape(prediction1,(imheight,imwidth))
    prediction2 = np.reshape(prediction2,(imheight,imwidth))
    
    
    op_img[hig-imheight:hig,0:int(wid/2)]=prediction1
    op_img[hig-imheight:hig,int(wid/2):wid]= prediction2 
    norm_image = 255*np.uint8(op_img) 
    print(tile)            

    cv2.imwrite(Path2write+tileNo[0]+'.png',norm_image)

print("Done with Images :)")
