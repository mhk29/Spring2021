import os
import tensorflow as tf
import nibabel as nib
import tensorflow.keras as keras
from tensorflow.keras import layers
from tqdm import tqdm
import numpy as np
import keyboard
import matplotlib.pyplot as plt
from math import floor
import math
from IPython.display import clear_output
from skimage.transform import resize

def dice_metric(y_true, y_pred):

    threshold = 0.5

    mask = y_pred > threshold
    mask = tf.cast(mask, dtype=tf.float32)
    y_pred = tf.multiply(y_pred, mask)
    mask = y_true > threshold
    mask = tf.cast(mask, dtype=tf.float32)
    y_true = tf.multiply(y_true, mask)
    # y_pred = tf.cast(y_pred > threshold, dtype=tf.float32)
    # y_true = tf.cast(y_true > threshold, dtype=tf.float32)

    inse = tf.reduce_sum(tf.multiply(y_pred, y_true))
    l = tf.reduce_sum(y_pred)
    r = tf.reduce_sum(y_true)

    # new haodong
    hard_dice = (2. * inse) / (l + r)

    hard_dice = tf.reduce_mean(hard_dice)

    return hard_dice


def get_names(img_path, mask_path):
    niftidir = os.listdir(img_path) 
    maskdir = os.listdir(mask_path) 
    niftinames = []
    masknames = []
    for i in niftidir:
        cname = format(i[0:6])
        for j in maskdir:
            niftinames.append(i)
            if (cname == format(j[0:6])) and (j not in masknames):
                masknames.append(j)
                break
            else:
                niftinames.remove(i)
    # print(niftinames)
    # print(masknames)
    if ".DS_Store" in niftinames:
        niftinames.remove(".DS_Store")
    if ".DS_Store" in masknames:
        masknames.remove(".DS_Store")
    return niftinames, masknames

def normalize(input_image):
    input_image = input_image.astype(float) 
    image_max = np.amax(input_image)
    image_min = np.amin(input_image)
    if image_max < 1.0 and image_min >= 0.0:
        return input_image

    input_image = (input_image - image_min) / ((image_max - image_min))
    if np.amin(input_image) != 0.0:
        input_image = (input_image - image_min) / ((1.0000000000000000000001)*(image_max - image_min))
    if np.amax(input_image) != 1.0:
        input_image = (input_image - image_min) / ((1.0000000000000000000001)*(image_max - image_min))

    # plt.figure(1)
    # plt.imshow(input_image[2,:,:,50])
    # plt.show()

    return input_image

def ceiling_norm(input_image):
    input_image = input_image.astype(float) 
    image_max = np.amax(input_image)
    image_min = np.amin(input_image)
    if image_max < 1.0 and image_min >= 0.0:
        return input_image

    input_image = (input_image - image_min) / ((image_max - image_min))
    if np.amin(input_image) != 0.0:
        input_image = (input_image - image_min) / ((1.0000000000000000000001)*(image_max - image_min))
    if np.amax(input_image) != 1.0:
        input_image = (input_image - image_min) / ((1.0000000000000000000001)*(image_max - image_min))

    input_image = np.ceil(input_image)

    plt.figure(1)
    plt.imshow(input_image[2,:,:,50])
    plt.show()

    return input_image


def multimaskload(image_height, image_width, image_depth, mask_path, train_mask):
    print("Loading Nifti Masks")
    Y_train = np.zeros((len(train_img), image_height, image_width, image_depth), dtype=np.uint16) 
    for n, id_ in tqdm(enumerate(train_mask), total=len(train_mask)):
        path = mask_path + id_
        img = np.array(nib.load(path).get_fdata())

        # Getting middle slices of a given mask of specified size
        Y_train[n,:,:,:] = img[floor(img.shape[0]/2)-floor(image_height/2):floor(img.shape[0]/2)-floor(image_height/2)+image_height,
                                floor(img.shape[1]/2)-floor(image_width/2):floor(img.shape[1]/2)-floor(image_width/2)+image_width,
                                floor(img.shape[2]/2)-floor(image_depth/2):floor(img.shape[2]/2)-floor(image_depth/2)+image_depth]



def load(image_height, image_width, image_depth, img_path, mask_path, train_img, train_mask):
    # Start with empty array to fill all images needed into the appropriate array
    X_train = np.zeros((len(train_img), image_height, image_width, image_depth), dtype=np.float32) 
    Y_train = np.zeros((len(train_img), image_height, image_width, image_depth), dtype=np.uint16) 

    print("Loading Nifti Images")
    for n, id_ in tqdm(enumerate(train_img), total=len(train_img)):
        path = img_path + id_
        img = np.array(nib.load(path).get_fdata())
        # Getting middle slices of a given image of specified size
        X_train[n,:,:,:] = img[floor(img.shape[0]/2)-floor(image_height/2):floor(img.shape[0]/2)-floor(image_height/2)+image_height,
                                floor(img.shape[1]/2)-floor(image_width/2):floor(img.shape[1]/2)-floor(image_width/2)+image_width,
                                floor(img.shape[2]/2)-floor(image_depth/2):floor(img.shape[2]/2)-floor(image_depth/2)+image_depth]
        # X_norm = X_resize

    print("Loading Nifti Masks")
    for n, id_ in tqdm(enumerate(train_mask), total=len(train_mask)):
        path = mask_path + id_
        img = np.array(nib.load(path).get_fdata())

        # Getting middle slices of a given mask of specified size
        Y_train[n,:,:,:] = img[floor(img.shape[0]/2)-floor(image_height/2):floor(img.shape[0]/2)-floor(image_height/2)+image_height,
                                floor(img.shape[1]/2)-floor(image_width/2):floor(img.shape[1]/2)-floor(image_width/2)+image_width,
                                floor(img.shape[2]/2)-floor(image_depth/2):floor(img.shape[2]/2)-floor(image_depth/2)+image_depth]
        # Y_norm = Y_resize
    X_norm = normalize(X_train)
    Y_norm = ceiling_norm(Y_train)



    return X_train, Y_norm


def valload(image_height, image_width, image_depth, img_path):
    train_img = os.listdir(img_path) 
    if ".DS_Store" in train_img:
        train_img.remove(".DS_Store")

    # Start with empty array to fill all images needed into the appropriate array
    X_train = np.zeros((len(train_img), image_height, image_width, image_depth), dtype=np.float32) # changed from uint8

    print("Loading Test Nifti Images")
    for n, id_ in tqdm(enumerate(train_img), total=len(train_img)):
        path = img_path + id_
        img = np.array(nib.load(path).get_fdata())
        # Getting middle slices of a given image of specified size
        X_train[n,:,:,:] = img[floor(img.shape[0]/2)-floor(image_height/2):floor(img.shape[0]/2)-floor(image_height/2)+image_height,
                                floor(img.shape[1]/2)-floor(image_width/2):floor(img.shape[1]/2)-floor(image_width/2)+image_width,
                                floor(img.shape[2]/2)-floor(image_depth/2):floor(img.shape[2]/2)-floor(image_depth/2)+image_depth]
        if n == 0:
            for x in range(img.shape[2]):
                dataOut = np.zeros((len(train_img)*X_train[0].shape[2], X_train[0].shape[0], X_train[0].shape[1], 1))
                dataOut[x,:,:,0] = X_train[0][:,:,x]
        else:
            for x in range(img.shape[2]):
                dataOut[x+n*x,:,:,0] = X_train[n][:,:,x]

    data_norm = normalize(dataOut)

    return data_norm


def fill(arrayData, arrayTruth):

    print("Filling Training ")

    for n, id_ in tqdm(enumerate(arrayData), total=len(arrayData)):
        if n == 0:
            for x in range(arrayData[0].shape[2]):
                dataOut = np.zeros((len(arrayData)*arrayData[0].shape[2], arrayData[0].shape[0], arrayData[0].shape[1], 1))
                truthOut = np.zeros((len(arrayTruth)*arrayTruth[0].shape[2], arrayTruth[0].shape[0], arrayTruth[0].shape[1], 1)) # changed from uint8
                dataOut[x,:,:,0] = arrayData[0,:,:,x]
                truthOut[x,:,:,0] = arrayTruth[0,:,:,x]
        else:
            for x in range(arrayData[0].shape[2]):
                dataOut[x+n*x,:,:,0] = arrayData[n,:,:,x]
                truthOut[x+n*x,:,:,0] = arrayTruth[n,:,:,x]

    return dataOut, truthOut

def predictions(model, data=None):

    # model.save('visModel')
    predictions = model.predict(data)
    # predictions = np.round(predictions)
    plt.imshow(2000*predictions[3*100+65, :, :, 0], cmap=plt.cm.gray)
    plt.show()

    return

# def display():

    # data, model original called parameters
    # # model.save('myModel')
    # predictions = model.predict(data[3:4])
    # predictions = np.round(predictions)

    # # plt.ion()
    # # plt.axis('off')

    # print(len(predictions[0][0][0]))

    # # plt.figure(0)
    # # plt.imshow(np.random.random((50,50)))
    # # plt.imshow(data[3, :, :, 0], cmap='gray')
    # plt.imshow(predictions[0, :, :, 0], alpha=0.2)
    # plt.show()
    # plt.pause(0.01)

    # # x = 0
    # # View = 0

    # plt.clf()
    # plt.imshow(data[3, :, :, x], cmap='gray')
    # plt.imshow(predictions[0, :, :, x], alpha=0.2)
    # plt.axis('off')
    # plt.draw()
    # plt.pause(0.01)

def unet_model(input_height, input_width, output_channels):

    input_layer = keras.layers.Input(shape=(input_height, input_width, 1))
    conv1a = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv1a = keras.layers.Dropout(0.2)(conv1a)
    conv1b = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv1a)
    pool1 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv1b)
    conv2a = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    conv2a = keras.layers.Dropout(0.2)(conv2a)
    conv2b = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv2a)
    pool2 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv2b)
    conv3a = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
    conv3a = keras.layers.Dropout(0.2)(conv3a)
    conv3b = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(conv3a)

    dconv3a = keras.layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), padding='same')(conv3b)
    dconv3a = keras.layers.Dropout(0.2)(dconv3a)
    dconv3b = keras.layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), padding='same')(dconv3a)
    unpool2 = keras.layers.UpSampling2D(size=(2, 2))(dconv3b)
    cat2 = keras.layers.concatenate([conv2b, unpool2])
    dconv2a = keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same')(cat2)
    dconv2a = keras.layers.Dropout(0.2)(dconv2a)
    dconv2b = keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same')(dconv2a)
    unpool1 = keras.layers.UpSampling2D(size=(2, 2))(dconv2b)
    cat1 = keras.layers.concatenate([conv1b, unpool1])
    dconv1a = keras.layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), padding='same')(cat1)
    dconv1a = keras.layers.Dropout(0.2)(dconv1a)
    dconv1b = keras.layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), padding='same')(dconv1a)

    output = keras.layers.Conv2D(filters=output_channels, kernel_size=(1, 1), activation='sigmoid', padding='same')(dconv1b)

    model = keras.models.Model(input_layer, output, name="u-netmodel")
    return model
