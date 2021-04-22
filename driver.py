import os
import tensorflow as tf
import nibabel as nib
import tensorflow.keras as keras
from tensorflow.keras import layers
from tqdm import tqdm
import numpy as np
import keyboard
import matplotlib.pyplot as plt

import vis

# these are the smallest common size among image set; got to be set manually 
IMG_WIDTH = 148
IMG_HEIGHT = 148
IMG_DEPTH = 100

img_path = "/Users/mattk/Documents/2020-2021/BME494/code20train/images/"
mask_path = "/Users/mattk/Documents/2020-2021/BME494/code20train/masks/"

val_path = "/Users/mattk/Documents/2020-2021/BME494/code10test/"

imagenames, masknames = vis.get_names(img_path, mask_path)
arrayData, arrayTruth = vis.load(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, img_path, mask_path, imagenames, masknames)
x, y = vis.fill(arrayData, arrayTruth)

valData = vis.valload(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, val_path)

# plt.figure(1)
# plt.imshow(arrayData[5,:,:,50])
# plt.show()

# note the use of sparse_categorical_crossentropy, 
# binary_crossentropy performs incorrect calculation for more than two labels
model = vis.unet_model(IMG_HEIGHT, IMG_WIDTH, 1)
opt = keras.optimizers.Adam()
model.compile(optimizer=opt, 
              loss='binary_crossentropy', 
              metrics=[keras.metrics.binary_accuracy, vis.dice_metric])

# x = np.rot90(x, axes=(2, 3))
# y = np.rot90(y, axes=(2, 3))
history = model.fit(x, y, 
                    steps_per_epoch=63, 
                    epochs=2, 
                    validation_data=valData.all()
                    # ,callbacks=[vis.DisplayCallback()] 
                    )

vis.predictions(model, valData)

# print(history.history['loss'])

# xvals = np.arange(2)
# plt.figure(1)
# plt.plot(xvals, history.history['binary_accuracy'])
# plt.savefig('plotA.png')
# np.savetxt("BA.csv", history.history['binary_accuracy'], delimiter=",")

# plt.figure(1)
# plt.plot(xvals, history.history['loss'])
# plt.savefig('plotL.png')
# np.savetxt("L.csv", history.history['loss'], delimiter=",")

# plt.figure(1)
# plt.plot(xvals, history.history['dice_metric'])
# plt.savefig('plotM.png')
# np.savetxt("M.csv", history.history['dice_metric'], delimiter=",")

# vis.display(arrayData, model)
