# -*- coding: utf-8 -*-
"""
Functions to integrate your model with the DEEPaaS API.
It's usually good practice to keep this file minimal, only performing the interfacing tasks.
In this way you don't mix your true code with DEEPaaS code and everything is more modular.
That is, if you need to write the predict() function in api.py, you would import your true predict
function and call it from here (with some processing/postprocessing in between if needed).
For example:

    import utils

    def predict(**kwargs):
        args = preprocess(kwargs)
        resp = utils.predict(args)
        resp = postprocess(resp)
        return resp

To start populating this file, take a look at the docs [1] and at a canonical exemplar module [2].

[1]: https://docs.deep-hybrid-datacloud.eu/
[2]: https://github.com/deephdc/demo_app
"""

from functools import wraps
from pathlib import Path
import pkg_resources

from aiohttp.web import HTTPBadRequest


BASE_DIR = Path(__file__).resolve().parents[1]


def _catch_error(f):
    """
    Decorate function to return an error as HTTPBadRequest,
    in case it fails.
    """
    @wraps(f)
    def wrap(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            raise HTTPBadRequest(reason=e)
    return wrap


@_catch_error
def get_metadata():
    """
    DO NOT REMOVE - All modules should have a get_metadata() function
    with appropriate keys.
    """
    distros = list(pkg_resources.find_distributions(str(BASE_DIR),
                                                    only=True))
    if len(distros) == 0:
        raise Exception('No package found.')
    pkg = distros[0]  # if several select first

    meta_fields = {'name', 'version', 'summary', 'home-page', 'author',
                   'author-email', 'license'}
    meta = {}
    for line in pkg.get_metadata_lines("PKG-INFO"):
        line_low = line.lower()  # to avoid inconsistency due to letter cases
        for k in meta_fields:
            if line_low.startswith(k + ":"):
                _, value = line.split(": ", 1)
                meta[k] = value

    return meta





#############################################################################################
########################### My work start from here #########################################
#############################################################################################



# Basic modules for api
from functools import wraps
import shutil
import tempfile
#
from aiohttp.web import HTTPBadRequest
from webargs import fields, validate

#import pickle
#import base64

########## Importing Other packages start from here 


# OS and others packages
import os
import sys

import numpy as np
import matplotlib.pyplot as plt


# image processing libraries

#import skimage.io as io
#import cv2

from PIL import Image
from skimage.io import imread, imsave

from skimage.color import rgb2gray

from skimage import transform
from skimage import img_as_bool


# Tensorflow packages 
import tensorflow as tf

from tensorflow.keras.models import Model, load_model

#from tensorflow import keras
#from tensorflow.keras import backend as K
   

from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
#from focal_loss import BinaryFocalLoss

import h5py
from tqdm import tqdm

#import zipfile
if sys.version_info >= (3, 6):
    import zipfile
else:
    import zipfile36 as zipfile
#import shutil
import zlib
from pathlib import Path



# Some parametres for preprocessing image_input

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3

input_size_2 = (IMG_WIDTH, IMG_HEIGHT)
input_size_3 = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)



## Metrics for prediction 

def dice_coefficient(y_true, y_pred):
    eps = 1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + eps) #eps pour éviter la division par 0 


def dice_malad(y_true, y_pred):
    epsilon = K.epsilon()
    y_true_f = K.flatten(y_true[:,:,:,2])
    y_pred_f = K.clip(K.flatten(y_pred[:,:,:,2]), epsilon, 1-epsilon)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f) + epsilon)



def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss



##############################################################################################################################
################################################ API functions start from here ############################################### 
##############################################################################################################################



#def _catch_error(f):
#    """Decorate function to return an error as HTTPBadRequest, in case
#    """
#    @wraps(f)
#    def wrap(*args, **kwargs):
#        try:
#            return f(*args, **kwargs)
#        except Exception as e:
#            raise HTTPBadRequest(reason=e)
#    return wrap


#def get_metadata():
#    metadata = {
#        "author": "Adnane"
#    }
#    return metadata




######################################################################################################
################################# Begin of Inference  ###############################################




def get_predict_args():
    """
    Input fields for the user.
    """
    arg_dict = {
        "demo-image": fields.Field(
            required=False,
            type="file",
            location="form",
            description="image",  # needed to be parsed by UI
        ),
        
        "data": fields.Field(
            description="Data file to perform inference on.",
            required=False,
            missing=None,
            type="file",
            location="form")
        ,
        # Add format type of the response of predict()
        # For demo purposes, we allow the user to receive back
        # either an image or a zip containing an image.
        # More options for MIME types: https://mimeapplication.net/
        "accept": fields.Str(
            description="Media type(s) that is/are acceptable for the response.",
            validate=validate.OneOf(["image/*", "application/zip"]),
        ),
    }
    return arg_dict
    




@_catch_error
def predict(**kwargs):
    """
    Return same inputs as provided.
    """
    #filepath = kwargs['demo-image'].filename
    #name = kwargs['demo-image'].name
    #content_type_info = kwargs['demo-image'].content_type
    #original_filename = kwargs['demo-image'].original_filename
    
    ## Import file
    #data = imread(filepath)


    # Return the result directly
    if kwargs['accept'] == 'image/*':
    
        ########## Preprocessing 

        filepath = kwargs['demo-image'].filename
        # 1 case , one image
        #read_image = data
        list_images = []

        read_image = imread(filepath)   
        image_resized = transform.resize(read_image, input_size_2).astype(np.float32)
        
        list_images.append(image_resized) 
        
        X_test_one = np.array(list_images)

        print("Shape of this image" , X_test_one.shape)
        print("Preprocessing Done")


        #load the best model
        path_ = os.getcwd()

        print("this is the path", path_)
        
        # me : change weights according to your uploading model    
        weights = np.array([1.0, 1.0, 1.0])
    
        load_model = tf.keras.models.load_model('./unet/unet/models_folder/best_model.h5', custom_objects={'loss':weighted_categorical_crossentropy(weights), 'dice_malad': dice_malad})
        #model_New = tf.keras.models.load_model('models/models for egi/best_model_all_data_update_loss_dice_copy2_val.h5',custom_objects={'loss':weighted_categorical_crossentropy(weights), 'dice_malad': dice_malad})

        
        print("Model : successfully loaded")


        # inference for image
        prediction = load_model.predict(X_test_one) #

        # change threshold depending to your loaded model
        preds_test_t = (prediction > 0.4).astype(np.uint8)

        X_test_one_result = np.squeeze(preds_test_t[0,:,:,2])*255

        print('inference Done')


        # Saving result 
        imsave("output.png", X_test_one_result)



        return open("output.png", 'rb')

    



    # Return a zip
    elif kwargs['accept'] == 'application/zip' :
        

        filepath2 = kwargs['data'].filename
        name = kwargs['data'].name
        content_type_info = kwargs['data'].content_type
        original_filename = kwargs['data'].original_filename
        
        zip_dir = tempfile.TemporaryDirectory()

        # Add original image to output zip
        #shutil.copyfile("output.png",
        #                zip_dir.name + '/demo.png')

        #shutil.copyfile(filepath,
        #        zip_dir.name + '/demo.png')        
        
        #shutil.copyfile(filepath2,
        #        zip_dir.name + '/demo')
        
        #shutil.copyfile(filepath2,
        #    zip_dir.name + '/' + original_filename)

        shutil.copyfile(filepath2,
            original_filename)

        #data_path = zip_dir.name + '/' + original_filename

        #with ZipFile(data_path, 'r') as obj_zip:
        #    FileNames = obj_zip.namelist()
        #    print("the list files in zip", FileNames)


        #with ZipFile(original_filename, 'r') as zipObj:
            #Extract all the contents of zip file in current directory
        #    zipObj.extractall()

        # Add for example a demo txt file
        with open(f'{zip_dir.name}/demo.txt', 'w') as f:
            f.write('Add here any additional information!')




        
        

        ####### Some information about input
        print(os.getcwd())
        print("THis is the name", name)
        print("THis is the content_type", content_type_info) 
        print("THis is the original_file_name", original_filename) 
        print("filepath2", filepath2)
    


        
        ########## Preprocessing 
        print("Preprocessing Start")

        # 2 case , dataset.zip

        # Extract dataset zip file

        with zipfile.ZipFile(original_filename, 'r') as zip_ref:
            zip_ref.extractall()
        
        print('Unpack the data zip file done')



        root = os.getcwd()
        file_path_0 = root + '/' + original_filename
        file_name_0 = Path(file_path_0).stem
        
        print("the root", root)
        print("the file_path_0", file_path_0)
        print("the file_name_0", file_name_0)
        #os.makedirs(root+data_dir_path, exist_ok=True)
        
        clean_files_path_0 = root + '/' + file_name_0
        #print("the os.listdir", os.listdir(clean_files_path_0))
        
        #print("This should be an output like this [images, masks] or this [masks, images] ")

        #path_to_images = clean_files_path_0 + "/images" 
        #path_to_masks = clean_files_path_0 + "/masks" 
        path_to_images = clean_files_path_0

        images_name = os.listdir(path_to_images)
        #masks_names = os.listdir(path_to_masks)

        #print("The names of images and maks must be the same and it's", images_names==masks_names)
        print(f"You have {len(images_name)} images in your dataset ")        


        #########
        # a revoir
        images_name = images_name
        path = path_to_images

        X_test_data = []

        # if train or test ....(i should add a funtion to precise , if train or test)
        for img_name in tqdm(images_name):
            # preprocessing images
            x_read_image = imread(path+'/'+img_name)
            x_image_resized = transform.resize(x_read_image, input_size_2).astype(np.float32)
            X_test_data.append(x_image_resized)
            
        X_test_data = np.array(X_test_data)


        print("Preprocessing data Done !")
        print("shape of your data", X_test_data.shape)


        ### Load Model 
        
        # me : change weights according to your uploading model    
        weights = np.array([1.0, 1.0, 1.0])
    
        load_model = tf.keras.models.load_model('./unet/unet/models_folder/best_model.h5', custom_objects={'loss':weighted_categorical_crossentropy(weights), 'dice_malad': dice_malad})
        #model_New = tf.keras.models.load_model('models/models for egi/best_model_all_data_update_loss_dice_copy2_val.h5',custom_objects={'loss':weighted_categorical_crossentropy(weights), 'dice_malad': dice_malad})

        print("Model : successfully loaded and ready to do prediction ")
        
        
        ### Inference for dataset 
        print("inference start ")

        prediction_data = x_load_model.predict(X_test_data)
        
        #threshold can be edited
        prediction_data_thresh = (prediction_data > 0.4).astype(np.uint8)

        print("Prediction done")
        

        ## Saving result 
        print("Saving results ... ")
        
        for ix, img_name in tqdm(zip(range(len(images_name)), images_name), total=len(images_name)): 
            result_mask = np.squeeze(prediction_data_thresh[ix,:,:,2])*255
            imsave(f'{zip_dir.name}' + '/' + f'{img_name}', result_mask)

        print('inference Done')
        print("You can dowload the masks, the temporary directory below")
        
        # Saving result into zip file
        # Pack dir into zip and return it
        shutil.make_archive(zip_dir.name, format='zip', root_dir=zip_dir.name)
        zip_path = zip_dir.name + '.zip'
        print("THis the zip path", zip_path)
        #print("new_ version")
        return open(zip_path, 'rb')





################################### End of Inference #######################################################
############################################################################################################







############################### Retraing Model Part ######################################################
##########################################################################################################


##### Import libraries, there is duplicate modules here, but there is no problem 

#step1: Load basic libraries
import numpy as np
import pandas as pd
from tqdm import tqdm

import os
import sys
import random
import warnings

#step2: Load libraries for the U-net Model
import tensorflow as tf


# Loading libraries for unet model start here.
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle

import random
# Make sure you have h5py, it used for saving and loading model
#!pip install h5py

#step3: Load other libraries
import skimage.io as io
import cv2
from skimage import transform
from skimage import img_as_bool
import matplotlib.pyplot as plt

###


#Set some parameters, Sizes and channel for images and masks:
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3

input_size_2 = (IMG_WIDTH, IMG_HEIGHT)
input_size_3 = (IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS)


######################################################################
########################### Preprocessing data ######################## 

# Define functions for pre-processing data
def custom_to_categorical(mask):
    M = np.zeros(np.hstack((mask.shape,3)))
    M[:,:,0] = (mask==0)
    M[:,:,1] = np.logical_and(mask>0,mask<255)
    M[:,:,2] = (mask==255)
    return img_as_bool(M).astype('int')#, M


# define paths for train and test data

train_path = "data/global data until 30_04/data_avec_export_label/train/"
test_path = "data/global data until 30_04/data_avec_export_label/test/"


##### Train set
path = train_path 

images_name = os.listdir(path+"images") # path for images
masks_name = os.listdir(path+"masks") # path for masks

#Shuffling data
random.Random(4000).shuffle(images_name)
random.Random(4000).shuffle(masks_name)


X_train = []
Y_train = []


for img_name, mask_name in tqdm(zip(images_name, masks_name), total=len(images_name)):
    # preprocessing images
    read_image = io.imread(path+"images/"+img_name)
    image_resized = transform.resize(read_image, input_size_2).astype(np.float32)
    X_train.append(image_resized)
    
    #preprocessing masks
    #read_mask = io.imread(path+"masks/"+mask_name)
    #convert_gray = np.asarray(cv2.cvtColor(read_mask, cv2.COLOR_BGR2GRAY), dtype="uint8")
    convert_gray = cv2.imread(path+"masks/"+mask_name, cv2.IMREAD_GRAYSCALE)
    convert_gray = np.asarray(convert_gray, dtype="uint8")
    mask_resized = cv2.resize(convert_gray, input_size_2, interpolation=cv2.INTER_NEAREST)
    mask_ = custom_to_categorical(mask_resized)
    
    Y_train.append(mask_)
    #Y_train_mask.append(np.asarray(M_, "uint8"))
    
N = len(Y_train)
print(f"\r {N}: Images founded in train set")


################# Test set
path = test_path

images_name = os.listdir(path+"images") # path for images
masks_name = os.listdir(path+"masks") # path for masks

X_test = []
Y_test = []

# if train or test ....(i should add a funtion to precise , if train or test)
for img_name, mask_name in tqdm(zip(images_name, masks_name), total=len(images_name)):
    # preprocessing images
    read_image = io.imread(path+"images/"+img_name)
    image_resized = transform.resize(read_image, input_size_2).astype(np.float32)
    X_test.append(image_resized)
    
    #preprocessing masks
    #read_mask = io.imread(path+"masks/"+mask_name)
    #convert_gray = np.asarray(cv2.cvtColor(read_mask, cv2.COLOR_BGR2GRAY), dtype="uint8")
    convert_gray = cv2.imread(path+"masks/"+mask_name, cv2.IMREAD_GRAYSCALE)
    convert_gray = np.asarray(convert_gray, dtype="uint8")
    mask_resized = cv2.resize(convert_gray, input_size_2, interpolation=cv2.INTER_NEAREST)
    mask_ = custom_to_categorical(mask_resized)
    
    Y_test.append(mask_)
    
N = len(Y_test)
print(f"\r {N}: Images founded in test set")


##############


# convert data (into an ndarray)
#train set
X_train = np.array(X_train)
Y_train = np.array(Y_train)

# test set
X_test = np.array(X_test)
Y_test = np.array(Y_test)



### Compute weights for loss

w0 = np.sum(Y_train,axis=(0,1,2))
weights = np.round(np.sum(w0)/w0).astype(int)

default_weights = np.array([1.0, 1.0, 1.0])


####### Split data for train and val.
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)




###### Defining some funtions for train part
###### Metrics to evaluate the model 

def dice_malad(y_true, y_pred):
    epsilon = K.epsilon()
    y_true_f = K.flatten(y_true[:,:,:,2])
    y_pred_f = K.clip(K.flatten(y_pred[:,:,:,2]), epsilon, 1-epsilon)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f) + epsilon)


def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss

#####################################


## Print the shapes of data to check

print("shape of X_train", X_train.shape)
print("shape of Y_train", Y_train.shape)
print("shape of X_val", X_val.shape)
print("shape of Y_val", Y_val.shape)

print("shape of X_test", X_test.shape)
print("shape of Y_test", Y_test.shape)

#### Convert before fit, to avoid, tensorflow tensor type error 

Y_train_n = Y_train.astype(np.float32)
Y_val_n = Y_val.astype(np.float32)
Y_test_n = Y_test.astype(np.float32)


################ End of preprocessing data ###################
###############################################################



##############################################################
#################### Building the model ######################

## We need to build the model, because we have to let the user to change some hyperparameters, and paste the weights to the model built  

## Build the model 

num_classes = 3

#Each block of u-net architecture consist of two Convolution layers
# These two layers are written in a function to make our code clean
def conv2d_block(input_tensor, n_filters, kernel_size=3):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal',
               padding="same")(input_tensor) # padding="valid"
    x = Activation("relu")(x)
    # second layer
    """x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), 
               padding="same")(x)
    x = Activation("relu")(x)"""
    return x



# The u-net architecture consists of contracting and expansive paths which
# shrink and expands the inout image respectivly. 
# Output image have the same size of input image
def get_unet(input_img, n_filters,kernel_size=3):
    # contracting path # encoder
    c1 = conv2d_block(input_img, n_filters=n_filters*4, kernel_size=kernel_size) #The first block of U-net
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = conv2d_block(p1, n_filters=n_filters*8, kernel_size=kernel_size)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = conv2d_block(p2, n_filters=n_filters*16, kernel_size=kernel_size)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = conv2d_block(p3, n_filters=n_filters*32, kernel_size=kernel_size)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    
    c5 = conv2d_block(p4, n_filters=n_filters*64, kernel_size=kernel_size) # last layer on encoding path 
    
    # expansive path # decoder
    u6 = Conv2DTranspose(n_filters*32, (3, 3), strides=(2, 2), padding='same') (c5) #upsampling included
    u6 = concatenate([u6, c4])
    c6 = conv2d_block(u6, n_filters=n_filters*32, kernel_size=kernel_size)

    u7 = Conv2DTranspose(n_filters*16, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = conv2d_block(u7, n_filters=n_filters*16, kernel_size=kernel_size)

    u8 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = conv2d_block(u8, n_filters=n_filters*8, kernel_size=kernel_size)

    u9 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = conv2d_block(u9, n_filters=n_filters*4, kernel_size=kernel_size)
    
    #outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    outputs = Conv2D(num_classes, (1, 1), activation = 'softmax') (c9)
    #conv10 = Conv2D(n_classes, 1, activation = 'softmax')(conv9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model


############################ End of building model code ##############
######################################################################



#####################################################################
################# Begin creation of the model ######################

# the array below, contains weights that used for train model 
weights = np.array([2.5, 1.5, 10.0])


### Creating the model 
input_img = Input((X_train.shape[1], X_train.shape[2], 3), name='img')

#me : num of filters can be adjusted
model = get_unet(input_img, n_filters= 2*4)

model.compile(optimizer='adam', loss = weighted_categorical_crossentropy(weights), metrics=[dice_malad])
model.summary()

# Add early stopping and model check points
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=20, verbose=1, mode='min', restore_best_weights=True) #Stop training when a monitored metric has stopped improving.
#reduce_lr = ReduceLROnPlateau(monitor='val_loss',
#                      factor=0.8,
#                      patience=15,
#                      mode='min')

# Save to temporary directory 
checkpoint_filepath = "temp_directory"+"best_model_n+1.h5"


Model_check = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_filepath, monitor="val_loss", verbose=1, save_best_only=True,
    save_weights_only=False, mode='auto') #Callback to save the Keras model or model weights at some frequency.



# Load the weights from the last best model     
model.load_weights("best_model.h5")



######################### End of the creation of the model ##############
#########################################################################


#########################################################################
######################## Start the training #############################

# same with results
results = model.fit(X_train,Y_train_n, validation_data=(X_val, Y_val_n),
                batch_size=16, epochs=512,
                callbacks=[early_stop, Model_check])





####### Thresholding part - Compare with the last model and check the score ######

# checking X_test shape
print("shape of X_test", X_test.shape)
print("shape of Y_test", Y_test.shape)

#
weights = "will be the same as the training input weights "
model_New = tf.keras.models.load_model('import from the temporary direct' + 'best_model_n+1.h5',custom_objects={'loss':weighted_categorical_crossentropy(weights), 'dice_malad': dice_malad})

# This evaluation test could used just in the case of Y_test is available
eval_test = model_New.evaluate(X_test,Y_test_n)

#print performances of the model on test images
print("Test Loss = " + str(eval_test[0]))
print("Test Metric = ",str(eval_test[1]))


######### Threshold 

Mask_valid_pred_int = model_New.predict(X_val, verbose=2) #

def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)



## compute Dice-score for a set of thresholds from (0.1 to 0.9 with a step of 0.1)
prob_thresh = [i*10**-1 for i in range(1,10)]
#perf=[] # define an empty array to store the computed F1-score for each threshold
perf_dice_malad = []
#perf_ALL=[]
perf_dice_malad_ALL= []
for r in tqdm(prob_thresh): # all th thrshold values
    #preds_bin = ((Mask_valid_pred_int> r) + 0 )
    preds_bin_dice_malad = ((Mask_valid_pred_int> r) + 0 )
    
    #preds_bin1=preds_bin[:,:,:,:]
    preds_bin1_dice_malad=preds_bin_dice_malad[:,:,:,2]
    
    #GTALL=Y_val[:,:,:,:]
    GTALL_malad = Y_val[:,:,:,2]
    for jj in range(len(GTALL_malad)): # all validation images
        #predmask=preds_bin1[jj,:,:]
        predmask_malad = preds_bin1_dice_malad[jj,:,:]

        #GT=GTALL[jj,:,:]
        GT_malad=GTALL_malad[jj,:,:]

        #l = GT.flatten()
        #l_malad = GT_malad.flatten()
        l_malad = GT_malad
        #p = predmask.flatten()
        #p_malad = predmask_malad.flatten()
        p_malad = predmask_malad
        #perf.append(f1_score(l, p)) # re invert the maps: cells: 1, back :0
        perf_dice_malad.append(dice_coef(l_malad, p_malad)) # re invert the maps: cells: 1, back :0

    #perf_ALL.append(np.mean(perf))
    perf_dice_malad_ALL.append(np.mean(perf_dice_malad))

    #perf=[]
    perf_dice_malad = []



# Twwo importatnt things here, the max of the score, and the threshold

max_dice_malad = max(perf_dice_malad_ALL)
op_thr_dice_malad = prob_thresh[np.array(perf_dice_malad_ALL).argmax()]  # Find the x value corresponding to the maximum y value


print (' Best threshold is:', op_thr_dice_malad, 'for dice_malad-score=', max_dice_malad)



#################################################################################

#################################################################################
########################## TBD ##################################################

## TBD : here is a few lines of code, to choose the best one 

last_dice_value = ...
last_threshold_value = ...

new_dice = max_dice_malad
new_threshold_value = op_thr_dice_malad


if new_dice > last_dice_value:
    replace the model_n, with_ the new one model_n+1 in_ inference, Also replace the threshold_value 

else:
    keep the model_n     
    
preds_test = model_New.predict(X_test, verbose=1)
preds_test_opt_malad = (preds_test > op_thr_dice_malad).astype('int')


####################### End of analyse and Training #########################
#############################################################################







def get_train_args():
    
    """
    Define your hyperparametres here, like 
    input image weith and heights
    
    batch size 
    epochs
    adam opt (i've choose it to be adaptative, )
    num_filters
    
    """



    return {} "return the hyperparameters for train"





def train(**kwargs):
    
    return "dictionnary of something"

















################################# End of retraining part ###################################################
#############################################################################################################





















#############################################################################################
#############################################################################################

# def warm():
#     pass
#
#
# def get_predict_args():
#     return {}
#
#
# @_catch_error
# def predict(**kwargs):
#     return None
#
#
# def get_train_args():
#     return {}
#
#
# def train(**kwargs):
#     return None


################################################################
# Some functions that are not mandatory but that can be useful #
# (you can remove this if you don't need them)                 #
################################################################


# def _fields_to_dict(fields_in):
#     """
#     Function to convert mashmallow fields to dict()
#     """
#     dict_out = {}
#
#     for key, val in fields_in.items():
#         param = {}
#         param['default'] = val.missing
#         param['type'] = type(val.missing)
#         if key == 'files' or key == 'urls':
#             param['type'] = str
#
#         val_help = val.metadata['description']
#         if 'enum' in val.metadata.keys():
#             val_help = "{}. Choices: {}".format(val_help,
#                                                 val.metadata['enum'])
#         param['help'] = val_help
#
#         try:
#             val_req = val.required
#         except:
#             val_req = False
#         param['required'] = val_req
#
#         dict_out[key] = param
#     return dict_out
