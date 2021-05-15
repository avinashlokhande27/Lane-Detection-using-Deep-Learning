'''
Created on 14-May-2021

@author: Avinash
'''

import numpy as np
import cv2
import sklearn.metrics as metrics
from PIL import Image
from pathlib import Path

LABEL_PATH = '/data/binary_lane_bdd/Images/val/val_labels_all/'
OUTPUT_PATH = '/data/binary_lane_bdd/Images/val/val_data_all/output_vgg1/'
NO_OF_CLASSES = 2

LABEL_SUFFIX = '.jpg'
OUTPUT_SUFFIX = '.png'


confusion_matrix = np.zeros([NO_OF_CLASSES, NO_OF_CLASSES])

output_file_name_list = Path(OUTPUT_PATH).glob('*' + OUTPUT_SUFFIX)

for output_file_name in output_file_name_list:
    print(output_file_name)
    
    ground_truth_name = str(Path(LABEL_PATH) / (str(output_file_name.name).split(OUTPUT_SUFFIX)[0] + LABEL_SUFFIX))
    output_image=Image.open(output_file_name)
    output_image=np.array(output_image)
    ground_truth=Image.open(ground_truth_name)
    ground_truth=(np.array(ground_truth)>200).astype('uint8')
    ground_truth=ground_truth[:,:,0]
    if ground_truth is None:
        continue
    
    print(np.unique(ground_truth))
    output_image = output_image/255
    print(np.unique(output_image))
    
    confusion_matrix += metrics.confusion_matrix(ground_truth.reshape(-1), output_image.reshape(-1), range(NO_OF_CLASSES))

total_predictions = np.sum(confusion_matrix)
mean_accuracy = mean_iou = mean_dice = 0
for class_id in range(0, NO_OF_CLASSES):
    tp = confusion_matrix[class_id, class_id]
    fp = np.sum(confusion_matrix[: class_id, class_id]) + np.sum(confusion_matrix[class_id + 1 :, class_id])
    fn = np.sum(confusion_matrix[class_id, : class_id]) + np.sum(confusion_matrix[class_id, class_id + 1 :])
    tn = total_predictions - tp - fp - fn
    
    accuracy = (tp + tn) / (tn + fn + tp + fp) 
    mean_accuracy += accuracy

    if ((tp + fp + fn) != 0):
        iou = (tp) / (tp + fp + fn)
        dice = (2 * tp) / (2 * tp + fp + fn)
        rec=tp/(tp+fn)
        spe=tn/(tn+fp)
    else:
        iou = 0.0
        dice = 0.0

    mean_iou += iou
    mean_dice += dice

    print("CLASS: {}: Accuracy: {}, IOU: {}, Dice: {},Specificity: {},Sensitivity: {}"
          .format(class_id, accuracy, iou, dice,spe,rec))

mean_accuracy = mean_accuracy / (NO_OF_CLASSES)
mean_iou = mean_iou / (NO_OF_CLASSES)
mean_dice = mean_dice / (NO_OF_CLASSES)
print("Mean Accuracy: {}, Mean IOU: {}, Mean Dice: {}".format(mean_accuracy, mean_iou, mean_dice))
