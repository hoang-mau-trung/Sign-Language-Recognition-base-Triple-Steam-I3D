
#----------------------------------------------IMPORT LIBRARY---------------------------------------------------
import argparse
import os
import cv2 
import numpy as np
import tensorflow as tf 
import warnings

from tensorflow.keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten, MaxPooling3D, 
        MaxPooling2D, LeakyReLU,BatchNormalization, Reshape, AveragePooling3D, GlobalAveragePooling3D)

from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_file
from tensorflow.keras import backend as K
from TwoStreamID3 import Inception_Inflated3d

#-------------------------------------------DEFINE PARAMETER-------------------------------------------------------
#-------------> parameters option
parser = argparse.ArgumentParser(description='Parameters of Two Stream Inflaten Conv3D')
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--shape_image', type=int,default=None)
parser.add_argument('--num_frame',type=int, default=None)
parser.add_argument('--nclass', type=int, default=24)
parser.add_argument('--freeze', type=int, default=0)
parser.add_argument('--mode', type=int, default=2)
parser.add_argument('--data_format', type=int, default=1)
args = parser.parse_args()

'''
batch : batch size of a epouch.
epouch: number of epouchs when trainning.
nclass: number class output.
data_format : image data format to use (1: channels_last or 0:channels_first).
mode: mode of image input (3: RGB, 1: GRAY, 2: OPT FLOW)
freeze: block layer. 0 = no, 1 = yes
'''

#---------------> parameters default 
min_frame_size =  64
default_frame_size =  128
default_num_frames = 10
min_num_frames =  5
'''
default_frame_size: default input frames(images) width/height for the model.
min_frame_size: minimum input frames(images) width/height accepted by the model.
default_num_frames: default input number of frames(images) for the model.
min_num_frames: minimum input number of frames accepted by the model.
'''
#-------------------------------------------HANDEL INPUT---------------------------------------------------

if args.data_format == 0:
    if args.mode not in {1,2,3}:
        warnings.warn('This model usually expects 1 ,2 or 3 input channels.'
                    + 'Options: 3: RGB, 1: GRAY, 2: OPT FLOW')
    else:
        input_shape = (args.mode, default_num_frames, default_frame_size, default_frame_size)
else:
    if args.mode not in {1,2,3}:
        warnings.warn('This model usually expects 1 ,2 or 3 input channels.'
                    + 'Options: 3: RGB, 1: GRAY, 2: OPT FLOW')
    else:
        input_shape = (default_num_frames, default_frame_size, default_frame_size,args.mode)

if args.shape_image is not None and args.shape_image < min_frame_size:
    warnings.warn('Input shape is too small. Input shape of frames must be at least : ' 
                    + str(min_frame_size))
elif args.shape_image is not None:
        if args.data_format == 0:
            input_shape = (args.mode, default_num_frames, args.shape_image, args.shape_image)
        else:
            input_shape = (default_num_frames, args.shape_image, args.shape_image,args.mode)
            
if args.num_frame is not None and args.num_frame < min_num_frames:
    warnings.warn('Input number of frames must be at least ' +
                                     str(min_num_frames) )
elif args.num_frame is not None:
    if args.data_format == 0:
            input_shape = (args.mode, args.num_frame, args.shape_image, args.shape_image)
    else:
            input_shape = (args.num_frame, args.shape_image, args.shape_image,args.mode)
print("Shape input: ",input_shape)
input_shape = Input(shape = input_shape)
#-------------------------------------------CREAT MODEL----------------------------------------------------
'''
model refer from model of QuoVadis,Action Recognition? A New Model and the Kinetics Dataset
https://arxiv.org/abs/1705.07750
'''
if args.mode == 3:
    mode =  'rgb'
else:
    mode =  'opt'
model = Inception_Inflated3d(input_shape=input_shape, dropout_prob=0.5, data_format=args.data_format,mode=mode, nclass=args.nclass )
model.summary()
print("Number of layer in model : ", len(model.layers))
model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy']) 
#-------------------------------------------LOAD DATA TO TRAINNING-----------------------------------------

print("[INFOR] : Starting loading data form disk...........")

# color: True RGB
# color: False OPT Flow RGB

from loadData import load
(X_train, y_train, X_test, y_test) = load(mode = 'opt')

print("[INFOR] : Loading Done !")

from sklearn.preprocessing import LabelBinarizer
label_binary = LabelBinarizer()
y_train = label_binary.fit_transform(y_train)
print(label_binary.classes_)

y_train = np.asanyarray(y_train)

y_test = label_binary.transform(y_test)
y_test = np.asanyarray(y_test)


# #------------------------------------------TRAINNING-------------------------------------------------------
print("[INFOR]: Trainning........")
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size = args.batch,
                        epochs = args.epoch, verbose=1, shuffle=True)


loss = history.history['loss']
val_loss = history.history['val_loss']
epouchs =  range(1, len(loss) + 1)

acc = history.history['acc']
val_acc =  history.history['val_acc']

epouchs =  range(1, len(loss) + 1)

plt.plot(epouchs, acc, 'b', label = 'Trainning accurarcy')
plt.plot(epouchs, val_acc, 'r', label = 'Validation accurarcy')
plt.title('Training and Validation Accurarcy')
plt.legend()

plt.figure()
plt.plot(epouchs, loss, 'b', label = 'Trainning loss')
plt.plot(epouchs, val_loss, 'r', label = 'Validation loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
# #-------------------------------------------EVALUATION MODEL-----------------------------------------------

print("[INFOR]: Calculating model accurary")
scores = model.evaluate(X_test, y_test)
print(f"Test Accurary: {scores[1] * 100}")

# #------------------------------------------SAVE MODEL------------------------------------------------------
print("[INFOR]: Saving model")
model.save('I3D_OPT.h5')