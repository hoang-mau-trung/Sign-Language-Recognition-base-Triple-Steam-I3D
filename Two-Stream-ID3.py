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
from utils import transVideotoImg3D

#-------------------------------------------DEFINE PARAMETER-------------------------------------------------------
#-------------> parameters option
parser = argparse.ArgumentParser(description='Parameters of Two Stream Inflaten Conv3D')
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--shape_image', type=int,default=None)
parser.add_argument('--num_frame',type=int, default=None)
parser.add_argument('--nclass', type=int, default=101)
parser.add_argument('--freeze', type=int, default=0)
parser.add_argument('--mode', type=int, default=3)
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
default_frame_size =  224
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


def conv3d_bn(x,filters,num_frames,num_row, num_col,padding='same', strides=(1, 1, 1),
        use_bias = False, use_activation_fn = True,use_bn = True,name=None):
    """Utility function to apply conv3d + BN.
        # Arguments
            x: input tensor.
            filters: filters in `Conv3D`.
            num_frames: frames (time depth) of the convolution kernel.
            num_row: height of the convolution kernel.
            num_col: width of the convolution kernel.
            padding: padding mode in `Conv3D`.
            strides: strides in `Conv3D`.
            use_bias: use bias or not  
            use_activation_fn: use an activation function or not.
            use_bn: use batch normalization or not.
            name: name of the ops; will become `name + '_conv'`
                for the convolution and `name + '_bn'` for the
                batch norm layer.
        # Returns
            Output tensor after applying `Conv3D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv3D(
        filters, 
        kernel_size = (num_frames, num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        name=conv_name)(x)

    if use_bn:
        if args.data_format == 0:
            bn_axis = 1
        else:
            bn_axis = 4
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)

    if use_activation_fn:
        x = Activation('relu', name=name)(x)

    return x

WEIGHTS_NAME = ['rgb_kinetics_only', 'flow_kinetics_only', 'rgb_imagenet_and_kinetics', 'flow_imagenet_and_kinetics']

# path to pretrained models with top (classification layer)
WEIGHTS_PATH = {
    'rgb_kinetics_only' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels.h5',
    'flow_kinetics_only' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels.h5',
    'rgb_imagenet_and_kinetics' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels.h5',
    'flow_imagenet_and_kinetics' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels.h5'
}

# path to pretrained models with no top (no classification layer)
WEIGHTS_PATH_NO_TOP = {
    'rgb_kinetics_only' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels_no_top.h5',
    'flow_kinetics_only' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels_no_top.h5',
    'rgb_imagenet_and_kinetics' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels_no_top.h5',
    'flow_imagenet_and_kinetics' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels_no_top.h5'
}

def Inception_Inflated3d(input_shape=None,dropout_prob=0.0,):
    if args.data_format == 0:
        channel_axis = 1
    else:
        channel_axis = 4
    # Downsampling via convolution (spatial and temporal)
    x = conv3d_bn(input_shape, 64, 7, 7, 7, strides=(2, 2, 2), padding='same', name='Conv3d_1a_7x7')

    # Downsampling (spatial only)
    x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same', name='MaxPool2d_2a_3x3')(x)
    x = conv3d_bn(x, 64, 1, 1, 1, strides=(1, 1, 1), padding='same', name='Conv3d_2b_1x1')
    x = conv3d_bn(x, 192, 3, 3, 3, strides=(1, 1, 1), padding='same', name='Conv3d_2c_3x3')

    # Downsampling (spatial only)
    x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same', name='MaxPool2d_3a_3x3')(x)

    # Mixed 3b
    branch_0 = conv3d_bn(x, 64, 1, 1, 1, padding='same', name='Conv3d_3b_0a_1x1')

    branch_1 = conv3d_bn(x, 96, 1, 1, 1, padding='same', name='Conv3d_3b_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 128, 3, 3, 3, padding='same', name='Conv3d_3b_1b_3x3')

    branch_2 = conv3d_bn(x, 16, 1, 1, 1, padding='same', name='Conv3d_3b_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 32, 3, 3, 3, padding='same', name='Conv3d_3b_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_3b_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 32, 1, 1, 1, padding='same', name='Conv3d_3b_3b_1x1')

    x = concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_3b')

    # Mixed 3c
    branch_0 = conv3d_bn(x, 128, 1, 1, 1, padding='same', name='Conv3d_3c_0a_1x1')

    branch_1 = conv3d_bn(x, 128, 1, 1, 1, padding='same', name='Conv3d_3c_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 192, 3, 3, 3, padding='same', name='Conv3d_3c_1b_3x3')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same', name='Conv3d_3c_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 96, 3, 3, 3, padding='same', name='Conv3d_3c_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_3c_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_3c_3b_1x1')

    x = concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_3c')


    # Downsampling (spatial and temporal)
    x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding='same', name='MaxPool2d_4a_3x3')(x)

    # Mixed 4b
    branch_0 = conv3d_bn(x, 192, 1, 1, 1, padding='same', name='Conv3d_4b_0a_1x1')

    branch_1 = conv3d_bn(x, 96, 1, 1, 1, padding='same', name='Conv3d_4b_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 208, 3, 3, 3, padding='same', name='Conv3d_4b_1b_3x3')

    branch_2 = conv3d_bn(x, 16, 1, 1, 1, padding='same', name='Conv3d_4b_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 48, 3, 3, 3, padding='same', name='Conv3d_4b_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4b_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_4b_3b_1x1')

    x = concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4b')

    # Mixed 4c
    branch_0 = conv3d_bn(x, 160, 1, 1, 1, padding='same', name='Conv3d_4c_0a_1x1')

    branch_1 = conv3d_bn(x, 112, 1, 1, 1, padding='same', name='Conv3d_4c_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 224, 3, 3, 3, padding='same', name='Conv3d_4c_1b_3x3')

    branch_2 = conv3d_bn(x, 24, 1, 1, 1, padding='same', name='Conv3d_4c_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 64, 3, 3, 3, padding='same', name='Conv3d_4c_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4c_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_4c_3b_1x1')

    x = concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4c')

    # Mixed 4d
    branch_0 = conv3d_bn(x, 128, 1, 1, 1, padding='same', name='Conv3d_4d_0a_1x1')

    branch_1 = conv3d_bn(x, 128, 1, 1, 1, padding='same', name='Conv3d_4d_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 256, 3, 3, 3, padding='same', name='Conv3d_4d_1b_3x3')

    branch_2 = conv3d_bn(x, 24, 1, 1, 1, padding='same', name='Conv3d_4d_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 64, 3, 3, 3, padding='same', name='Conv3d_4d_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4d_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_4d_3b_1x1')

    x = concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4d')

    # Mixed 4e
    branch_0 = conv3d_bn(x, 112, 1, 1, 1, padding='same', name='Conv3d_4e_0a_1x1')

    branch_1 = conv3d_bn(x, 144, 1, 1, 1, padding='same', name='Conv3d_4e_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 288, 3, 3, 3, padding='same', name='Conv3d_4e_1b_3x3')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same', name='Conv3d_4e_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 64, 3, 3, 3, padding='same', name='Conv3d_4e_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4e_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_4e_3b_1x1')

    x = concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4e')

    # Mixed 4f
    branch_0 = conv3d_bn(x, 256, 1, 1, 1, padding='same', name='Conv3d_4f_0a_1x1')

    branch_1 = conv3d_bn(x, 160, 1, 1, 1, padding='same', name='Conv3d_4f_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 320, 3, 3, 3, padding='same', name='Conv3d_4f_1b_3x3')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same', name='Conv3d_4f_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 128, 3, 3, 3, padding='same', name='Conv3d_4f_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4f_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 128, 1, 1, 1, padding='same', name='Conv3d_4f_3b_1x1')

    x = concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4f')


    # Downsampling (spatial and temporal)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same', name='MaxPool2d_5a_2x2')(x)

    # Mixed 5b
    branch_0 = conv3d_bn(x, 256, 1, 1, 1, padding='same', name='Conv3d_5b_0a_1x1')

    branch_1 = conv3d_bn(x, 160, 1, 1, 1, padding='same', name='Conv3d_5b_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 320, 3, 3, 3, padding='same', name='Conv3d_5b_1b_3x3')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same', name='Conv3d_5b_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 128, 3, 3, 3, padding='same', name='Conv3d_5b_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_5b_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 128, 1, 1, 1, padding='same', name='Conv3d_5b_3b_1x1')

    x = concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_5b')

    # Mixed 5c
    branch_0 = conv3d_bn(x, 384, 1, 1, 1, padding='same', name='Conv3d_5c_0a_1x1')

    branch_1 = conv3d_bn(x, 192, 1, 1, 1, padding='same', name='Conv3d_5c_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 384, 3, 3, 3, padding='same', name='Conv3d_5c_1b_3x3')

    branch_2 = conv3d_bn(x, 48, 1, 1, 1, padding='same', name='Conv3d_5c_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 128, 3, 3, 3, padding='same', name='Conv3d_5c_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_5c_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 128, 1, 1, 1, padding='same', name='Conv3d_5c_3b_1x1')

    x = concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_5c')
    # x = AveragePooling3D((2, 7, 7), strides=(1, 1, 1), padding='valid', name='global_avg_pool')(x)
    # x = Dropout(dropout_prob)(x)

    # x = conv3d_bn(x, args.nclass, 1, 1, 1, padding='same', 
    #             use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1')
    model = Model(input_shape, x, name='i3d_inception')           
    # load weights
    weights_url = WEIGHTS_PATH_NO_TOP['rgb_imagenet_and_kinetics']
    model_name = 'i3d_inception_rgb_imagenet_and_kinetics_no_top.h5'
    downloaded_weights_path = get_file(model_name, weights_url, cache_subdir='models')
    model.load_weights(downloaded_weights_path)
    #block layer 
    if args.freeze == 1:
        for layer in model.layers:
            layer.trainable = False
        print("Block :" ,len(model.layers), " layers can not trainning!")


    # add ouput ------------------------------------------------------------------------------------

    x = model.output
    x = AveragePooling3D((2, 7, 7), strides=(1, 1, 1), padding='valid', name='global_avg_pool')(x)
    x = Dropout(dropout_prob)(x)

    x = conv3d_bn(x, args.nclass, 1, 1, 1, padding='same', 
                 use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1')
    x = Flatten()(x)
    x = Dense(args.nclass, activation='softmax')(x)
    model = Model(input_shape, x, name='i3d_ucf')
    return model
model = Inception_Inflated3d(input_shape=input_shape, dropout_prob=0.5)
model.summary()
print("Number of layer in model : ", len(model.layers))
model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy']) 
#-------------------------------------------LOAD DATA TO TRAINNING-----------------------------------------

print("[INFOR] : Starting loading data form disk...........")

# color: True RGB
# color: False OPT Flow RGB

(X,Y) = transVideotoImg3D.TrimVideo(color = True, nclass = args.nclass)

print("[INFOR] : Loading Done !")

from sklearn.preprocessing import LabelBinarizer

label_binary = LabelBinarizer()
y = label_binary.fit_transform(Y)

print(label_binary.classes_)

args.nclass = len(label_binary.classes_)

y = np.asanyarray(y)

# division data

from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2 ,random_state=42)


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
model.save('I3D_model.h5')