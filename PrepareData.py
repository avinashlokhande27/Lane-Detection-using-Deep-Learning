'''
Created on 14-May-2021

@author: Avinash
'''

import glob
import numpy as np
import cv2
import os
import albumentations as A

# Define Size of images
imwidth = 640
imheight = 544
imdepth = 3

# Number Of classes in labels
classes = 2
data_shape = imwidth*imheight

# Path to read Tiles for Label and Data
data_path = '/data/binary_lane_bdd/Images/train/prepro/crop_tiles/'
label_path = '/data/binary_lane_bdd/Images/train/prepro/crop_tiles/Label/'
objects_dir = '/data/binary_lane_bdd/Images/train/prepro/crop_tiles/obj_255_Aug_cat/'
data = []
label = []

# Function to create label array for binary classification
def binarylab(labels):
    
    # Define an Empty Array
    x = np.zeros([imheight, imwidth, classes], dtype="uint8")
    
    # Read Each pixel label and put it into corresponding label plane
    for i in range(imheight):
        for j in range(imwidth):
            x[i, j, labels[i][j]] = 1
    
    label.append(x)


def data_norm(im):
            
    im = im.astype("float32")              
    im /= 255
    data.append(np.rollaxis((im),2))


aug = A.Compose([
    A.RandomSizedCrop(min_max_height=(50, 101), height=544, width=640, p=0.5),
    A.CLAHE(p=0.8),
    A.RandomBrightnessContrast(p=0.2),
    A.HueSaturationValue(p=1),   
    A.RandomGamma(p=0.8)])
        
labelpaths = sorted(glob.glob(label_path + "/*.png"))

# Create Empty Lists to store Image and Label Data
for i in range(len(labelpaths)):
    tlp = labelpaths[i]
    tilename = os.path.basename(tlp)
    tdp = data_path+tilename[:-4]+'.jpg'
    
    # Read Images
    im = cv2.imread(tdp)
    lab = (cv2.imread(tlp)[:, :, 0] > 200).astype("uint8")
    
    if im is not None:
        augmented = aug(image=im, mask=lab)
        image = augmented['image']
        labe = augmented['mask']

        data_norm(im)
        data_norm(image)           
        # Convert label into binary form
        binarylab(lab)
        binarylab(labe)

        
        print('\n'+tdp)
    else:
        print("error: "+tdp)


os.makedirs(objects_dir, exist_ok=True)

np.savez_compressed(objects_dir+'train_data.npz', data)
np.savez_compressed(objects_dir+'train_data_labels.npz', label)
print("Done")

