from keras.callbacks import ModelCheckpoint, TensorBoard

from keras import backend as K
from weighted_hausdorff_loss import Weighted_Hausdorff_loss

import LoadBatches

from Models import FCN8, FCN32, SegNet, UNet, lab3model
from keras import optimizers
import math

#############################################################################
#train_images_path = "data/dataset1/images_prepped_train/"
#train_segs_path = "data/dataset1/annotations_prepped_train/"
train_images_path = "WARWICK/train/"
train_segs_path = "WARWICK/train_label/"

train_batch_size = 8
#n_classes = 11
n_classes = 2

epochs = 1

#input_height = 320
#input_width = 320

input_height = 128
input_width = 128


#val_images_path = "data/dataset1/images_prepped_test/"
#val_segs_path = "data/dataset1/annotations_prepped_test/"
val_images_path = "WARWICK/test/"
val_segs_path = "WARWICK/test_label/"

val_batch_size = 8

key = "lab3model"


##################################

method = {
    "fcn32": FCN32.FCN32,
    "fcn8": FCN8.FCN8,
    'segnet': SegNet.SegNet,
    'unet': UNet.UNet,
    'lab3model': lab3model.mnistmodel}

m = method[key](n_classes, input_height=input_height, input_width=input_width)

smooth =1
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

m.compile(optimizer="adam", loss=dice_coef_loss, metrics=[dice_coef, 'acc'])


#m.compile(optimizer="adam", loss=dice_coef_loss, metrics=['acc', Weighted_Hausdorff_loss])

#m.compile(loss= 'categorical_crossentropy', optimizer="adam", metrics=['acc'])

m.summary()



G = LoadBatches.imageSegmentationGenerator(train_images_path,
                                           train_segs_path, train_batch_size, n_classes=n_classes, input_height=input_height, input_width=input_width)

G_test = LoadBatches.imageSegmentationGenerator(val_images_path,
                                                val_segs_path, val_batch_size, n_classes=n_classes, input_height=input_height, input_width=input_width)

checkpoint = ModelCheckpoint(
    filepath="output/%s_model.h5" %
    key,
    monitor='acc',
    mode='auto',
    save_best_only='True')
#tensorboard = TensorBoard(log_dir='output/log_%s_model' % key)

m.fit_generator(generator=G,
                steps_per_epoch=math.ceil(367. / train_batch_size),
                epochs=epochs, callbacks=[checkpoint],
                verbose=1,
                validation_data=G_test,
                validation_steps=8,
                shuffle=True)
