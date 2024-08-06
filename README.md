# Lung Nodule Classification

This project aims to classify lung nodules into benign and malignant categories using a convolutional neural network (CNN) model. The model is trained on a dataset of lung nodule images and can be used to predict the class of new lung nodule images.

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Setup and Installation](#setup-and-installation)
5. [Training the Model](#training-the-model)
6. [Using the Model](#using-the-model)
7. [Results](#results)
8. [License](#license)

## Overview
The project involves:
- Loading and preprocessing the dataset.
- Defining and training a CNN model.
- Saving and loading the trained model.
- Predicting the class of new images.

## Dataset
The dataset used for this project contains lung nodule images categorized as benign or malignant. The dataset is divided into two directories:
- `Dataset A` for training data
- `Dataset B` for validation data

## Model Architecture
The CNN model consists of:
- Three convolutional layers with ReLU activation and max pooling.
- A flatten layer.
- A dense layer with 256 units and ReLU activation.
- An output dense layer with softmax activation for binary classification.

## Setup and Installation
1. Mount your Google Drive to access the dataset:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')

Install the required packages:

pip install tensorflow matplotlib pillow ipywidgets

Define data generators for training and validation datasets:
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = '/content/drive/MyDrive/Datasets/Dataset A'
val_dir = '/content/drive/MyDrive/Datasets/Dataset B'

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_ds = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), color_mode='grayscale')

val_datagen = ImageDataGenerator(rescale=1./255)
val_ds = val_datagen.flow_from_directory(val_dir, target_size=(224, 224), color_mode='grayscale')


Define and compile the model:
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

input_shape = (224, 224, 1)
input1 = Input(shape=input_shape)
conv1_1 = Conv2D(32, (3, 3), activation='relu')(input1)
maxpool1_1 = MaxPooling2D((2, 2))(conv1_1)
conv1_2 = Conv2D(64, (3, 3), activation='relu')(maxpool1_1)
maxpool1_2 = MaxPooling2D((2, 2))(conv1_2)
conv1_3 = Conv2D(128, (3, 3), activation='relu')(maxpool1_2)
maxpool1_3 = MaxPooling2D((2, 2))(conv1_3)
flatten1 = Flatten()(maxpool1_3)
dense1 = Dense(256, activation='relu')(flatten1)
output1 = Dense(2, activation='softmax')(dense1)

Train the model:


model1 = Model(inputs=input1, outputs=output1)
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

Train the model:
model1.fit(train_ds, epochs=10, validation_data=val_ds)
model1.save('lung_nodule.h5')

Load the trained model and create a file upload widget:
from tensorflow.keras.models import load_model
from PIL import Image
from IPython.display import display
from ipywidgets import FileUpload
import io
import matplotlib.pyplot as plt
import numpy as np

model = load_model('lung_nodule.h5')
upload = FileUpload()
display(upload)

Define a function to process and display the uploaded image:
def process_and_display_uploaded_image(change):
    uploaded_data = next(iter(upload.value.values()))
    uploaded_image = Image.open(io.BytesIO(uploaded_data['content']))
    analyze_image(uploaded_image)

upload.observe(process_and_display_uploaded_image, names='value')

def analyze_image(img):
    img = img.resize((224, 224)).convert('L')
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    class_idx = np.argmax(preds[0])
    class_label = ['benign', 'malignant'][class_idx]
    accuracy = preds[0][class_idx]

    plt.figure(figsize=(4, 2))
    plt.imshow(img, cmap='gray')
    plt.title(f'Prediction: {class_label}')
    plt.show()
    print(f'Accuracy: {accuracy:.2f}')
