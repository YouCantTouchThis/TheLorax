#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import os
import sklearn
import tensorflow as tf
import numpy as np
import shutil

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# In[4]:


from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator


# In[45]:


train = tf.keras.preprocessing.image_dataset_from_directory(
    "/Users/gaurishlakhanpal/Downloads/leave_set",
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="grayscale",
    batch_size=32,
    image_size=(64, 64),
    shuffle=True,
    seed=69,
    subset="training",
    validation_split = .2,
    interpolation="bilinear",
    follow_links=False,
)
test = tf.keras.preprocessing.image_dataset_from_directory(
    "/Users/gaurishlakhanpal/Downloads/leave_set",
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="grayscale",
    batch_size=32,
    image_size=(64, 64),
    shuffle=True,
    seed=69,
    validation_split=.2,
    subset="validation",
    interpolation="bilinear",
    follow_links=False,
)


# In[28]:


class_names = train.class_names
print(class_names)


# In[51]:


train.file_paths
print(train.file_paths[0][43:])
"""for x in (class_names):
    path = "/Users/gaurishlakhanpal/Downloads/leave_set1" + "/" + x
    try:
        os.makedirs(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)"""
        
for x in train.file_paths:
    
    shutil.move(x, '/Users/gaurishlakhanpal/Downloads/leave_set1' + x[43:])


# In[52]:


test.file_paths
print(test.file_paths[0][43:])
for x in (class_names):
    path = "/Users/gaurishlakhanpal/Downloads/leave_set2" + "/" + x
    try:
        os.makedirs(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)
        
for x in test.file_paths:
    
    shutil.move(x, '/Users/gaurishlakhanpal/Downloads/leave_set2' + x[43:])


# In[74]:


train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = .3,
        zoom_range = .3,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip = True)

train = train_datagen.flow_from_directory(
        '/Users/gaurishlakhanpal/Downloads/leave_set1',
        target_size = (64, 64),
        batch_size = 32,
        class_mode = 'categorical')

test_datagen = ImageDataGenerator(
        rescale = 1./255)

test = test_datagen.flow_from_directory(
        '/Users/gaurishlakhanpal/Downloads/leave_set2',
        target_size = (64, 64),
        batch_size = 32,
        class_mode = 'categorical')


# In[23]:


normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)


# In[79]:


num_classes = 8

model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(64, 64, 3)),
  layers.Conv2D(1, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(2, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(2, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(1, activation='softmax'),
  layers.Dense(num_classes)
])


# In[80]:


model.compile(optimizer='rmsprop',
              loss="categorical_crossentropy",
              metrics=['accuracy'])


# In[81]:


model.summary()


# In[82]:


epochs = 1
history = model.fit(
  train,
  validation_data=test,
  batch_size = 32,
  epochs=epochs
)


# In[16]:


checkpoint_path = "/Users/gaurishlakhanpal/desktop/training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


# In[17]:


tf.keras.models.save_model(
    model, checkpoint_path, overwrite=True, include_optimizer=True, save_format=None,
    signatures=None, options=None, save_traces=True
)


# In[11]:


get_ipython().system('pip install tensorflowjs')
import tensorflowjs as tfjs


# In[12]:


tfjs.converters.save_keras_model(model, "/Users/gaurishlakhanpal/desktop/training_2")


# In[85]:


model.save("/Users/gaurishlakhanpal/desktop/training_5")


# In[ ]:




