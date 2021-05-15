'''
Created on 14-May-2021

@author: Avinash
'''

import glob
import numpy as np
import cv2
from cv2 import cvtColor
import os
import matplotlib.pyplot as plt

img_path = '/data/binary_lane_bdd/Images/train/train_data_all/'
label_path = '/data/binary_lane_bdd/Images/train/train_labels_all/'
path2save = '/data/binary_lane_bdd/Images/train/prepro/'


os.makedirs(path2save, exist_ok=True)
os.makedirs(path2save+'crop/Label/', exist_ok=True)
os.makedirs(path2save+'crop_tiles/Label/', exist_ok=True)
    
def crop_tiling(img):
    
    hig,wid,dep = img.shape
    
    img_1=img[176:hig,0:int(wid/2)]
    img_2=img[176:hig,int(wid/2):wid]
    img_3=img[176:hig,:]
    
    return img_1,img_2,img_3
    
  
label_imgs = sorted(glob.glob(label_path+"/*.jpg"))
for lab_img in label_imgs:
    lab=cv2.imread(lab_img)
    lab=(lab>150).astype('uint8')
    
    lab_name=os.path.basename(lab_img)
    img=cv2.imread(img_path+lab_name)
    
    lab_1,lab_2,lab_3=crop_tiling(lab)
    img_1,img_2,img_3=crop_tiling(img)

    cv2.imwrite(path2save+'crop_tiles/Label/'+lab_name[:-4]+'_0.png',lab_1*255)
    cv2.imwrite(path2save+'crop_tiles/Label/'+lab_name[:-4]+'_1.png',lab_2*255)
    cv2.imwrite(path2save+'crop/Label/'+lab_name[:-4]+'.png',lab_3*255)
    
    cv2.imwrite(path2save+'crop_tiles/'+lab_name[:-4]+'_0.jpg',img_1)
    cv2.imwrite(path2save+'crop_tiles/'+lab_name[:-4]+'_1.jpg',img_2)
    cv2.imwrite(path2save+'crop/'+lab_name[:-4]+'.jpg',img_3)
    
    print(lab_name)

print('done')    
    
    
    
