## Lane-Detection-using-Deep-Learning


**Approach**

To implement deep learning algorithm to detect lanes on the street, I have used image segmentation approach. Now, to build a segmentation model need to follow following steps:
1.	Data pre-processing
2.	Augmentation
3.	Model selection and training
4.	Inferencing
5.	Model evaluation

**Challenges**

1.	When I got the assignment, I downloaded the data set, resized images to 384X640, created train-test split, used random augmentation and kept it for training got good training accuracy but when inferenced on test set got blank output.
2.	There are lot of segmentation model architectures and modified versions present, which one to choose?
3.	Noisy dataset with class imbalance


**Data pre-processing**

From total 501 images of size(1280x720), used 451 for training and 50 for testing. 

The narrow width lanes annotations might get distorted with image resizing. So, I have used cropping and tiling approach to pre-process the data.

When I gone through the data, labels were present only in bottom 75% of the image other 25% mostly content sky which was not that much useful in training and was increasing background data

Cropping sky region and splitting the remaining image in 2 tiles gives me 902 images of size 544x640 for training.

**Augmentation**

Following are the transformations I chose for augmentation with _‘Albumentations’_ python library
1.	Random sized crop
2.	Random Hue saturation
3.	Random brightness contrast
4.	Random gamma
5.	CLAHE(Contrast Limited Adaptive Histogram Equalization)

These augmentations can take care of different lighting conditions and camera specifications. These 5 transformations will increase the data set significantly

**Model selection and training**

From a large pool of network architectures, we usually prefer the most recent deep learning model, which might be unnecessary for the task at hand. Theoretically the choice of model depends on the properties of the dataset and the task. 
Instead of going for more complex network, I have used a simple straight forward sequential network VGG-FCN to solve this problem. which is easy to debug and always gives results

For training this model I have used Adam optimiser with initial learning rate 10e-4 and categorical crossentropy as loss function. 
Trained on NVIDIA V100 GPU with batch size 8 and for 500 epochs

**Inferencing**

For inferencing, cropped and tiled input test image as per pre-processing step. Carried out model prediction on both tiles and put them back together as original dimensions

**Model evaluation**

Model was evaluated on 50 test images based on IOU and Dice

Mean Accuracy: 0.981916015625, Mean IOU: 0.5997834993422672, Mean Dice: 0.6742227687229821
