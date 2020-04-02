from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.applications import mobilenet_v2
from keras import models
from keras import layers
from keras import optimizers

mobnet_conv = mobilenet_v2.MobileNetV2(input_shape=None, alpha=1.0, include_top=False, weights='imagenet', input_tensor=None, pooling=None)

train_dir = './clean-dataset/train'
validation_dir = './clean-dataset/validation'

nTrain = 3840
# nTrain = 6
nVal = 960
nImages = 4800

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 2

train_features = np.zeros(shape=(nTrain, 7, 7, 1280))
train_labels = np.zeros(shape=(nTrain,1))
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical')

i = 0
for inputs_batch, labels_batch in train_generator:
    features_batch = mobnet_conv.predict(inputs_batch)
    train_features[i * batch_size : (i + 1) * batch_size] = features_batch
    train_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
    i += 1
    print(i)
    if i * batch_size >= nTrain:
        break

train_features = np.reshape(train_features, (nTrain, 7 * 7 * 1280))

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=7 * 7 * 1280))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='softmax'))
