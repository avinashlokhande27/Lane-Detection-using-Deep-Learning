'''
Created on 14-May-2021

@author: Avinash
'''

from keras.callbacks import LearningRateScheduler,TensorBoard,ModelCheckpoint
from keras.optimizers import adam
import NetPixel
import numpy as np
from keras.backend.tensorflow_backend import set_session
import keras.backend as K
K.set_image_data_format('channels_first')


data_path = '/mnt/store01/Avinash/prostate_NHS/Lane-Detection-using-Deep-Learning-master/data/binary_lane_bdd/Images/train/prepro/crop_tiles/obj_255_Aug_cat/'
log_dir = '/mnt/store01/Avinash/prostate_NHS/Lane-Detection-using-Deep-Learning-master/data/binary_lane_bdd/Models/tf19_dp/logs/'
path2save = '/mnt/store01/Avinash/prostate_NHS/Lane-Detection-using-Deep-Learning-master/data/binary_lane_bdd/Models/tf19_dp/'

imheight = 544
imwidth = 640
imdepth = 3
data_shape = imheight*imwidth
classes = 2

train_data = np.load(data_path+'train_data.npz')['arr_0']
train_data  = train_data .astype("float32")

train_label = np.load(data_path+'train_data_labels.npz')['arr_0']
train_label = np.reshape(train_label,(len(train_label),data_shape,classes))

model = NetPixel.VGG_FCN(imwidth, imheight, imdepth, classes)
model.summary()

tbCallBack = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)

def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)


def lr_schedule(epoch):

    lr = 1e-2
    if epoch >= 50 and epoch < 100:
        lr *= 1e-1
    elif epoch >= 100 and epoch <150:
        lr *= 1e-2
    elif epoch >= 150 and epoch <200:
        lr *= 1e-3
    elif epoch >= 200 and epoch <500:
        lr *= 1e-4
    elif epoch >= 500:
        lr *= 1e-5
    print('Learning rate: ', lr)
    return lr


# Store the network every 5 epoch
filepath=path2save+'tf19_dp-{epoch:02d}-{val_loss:02f}-{val_dice_coef:.2f}-{val_acc:.2f}.h5'

modelCheck = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=False, save_weights_only= False, mode='min', period=5)
opt = adam(lr=lr_schedule(0))

print ("Compiling Model...")

# Set the compiler parameter for the training
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=[dice_coef,"accuracy"], sample_weight_mode='auto')

lr_scheduler = LearningRateScheduler(lr_schedule)

print ("Training the Model...")

model.fit(train_data, train_label, batch_size= 8, epochs = 1000, verbose=2,callbacks=[modelCheck,lr_scheduler, tbCallBack],validation_split=0.2)

