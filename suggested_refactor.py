from keras.preprocessing.image import ImageDataGenerator, load_img
from matplotlib import pyplot as plt
import numpy as np
from keras.applications import mobilenet_v2
from keras import models
from keras import layers
from keras import optimizers

# Set directory for training and evaluation dataset 
train_dir = './clean-dataset/train'
validation_dir = './clean-dataset/validation'

mobnet_conv = mobilenet_v2.MobileNetV2(input_shape=None, alpha=1.0, include_top=False, weights='imagenet', input_tensor=None, pooling=None)

# nTrain = 3840
nTrain = 20
# nVal = 960
nVal = 20

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

train_features = np.zeros(shape=(nTrain, 7, 7, 1280))
train_labels = np.zeros(shape=(nTrain,2))

validation_features = np.zeros(shape=(nVal, 7, 7, 1280))
validation_labels = np.zeros(shape=(nVal,2))

def trainmethod(train_features, train_labels):
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
      if i * batch_size >= nTrain:
          break

  train_features = np.reshape(train_features, (nTrain, 7 * 7 * 1280))

def evalmethod(validation_features, validation_labels):

  validation_generator = datagen.flow_from_directory(
      validation_dir,
      target_size=(224, 224),
      batch_size=batch_size,
      class_mode='categorical')

  i = 0
  for inputs_batch, labels_batch in validation_generator:
      features_batch = mobnet_conv.predict(inputs_batch)
      validation_features[i * batch_size : (i + 1) * batch_size] = features_batch
      validation_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
      i += 1
      if i * batch_size >= nVal:
          break

  validation_features = np.reshape(validation_features, (nVal, 7 * 7 * 1280))

def model():
  model = models.Sequential()
  model.add(layers.Dense(256, activation='relu', input_dim=7 * 7 * 1280))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(2, activation='softmax'))

  model.compile(optimizer=optimizers.RMSprop(lr=2e-4),
                loss='categorical_crossentropy',
                metrics=['acc'])


# print(train_labels)
# print(validation_labels)

  history = model.fit(train_features,
                      train_labels,
                      epochs=1,
                      batch_size=batch_size,
                      validation_data=
  (validation_features,validation_labels))

  fnames = validation_generator.filenames

  ground_truth = validation_generator.classes

  label2index = validation_generator.class_indices

  # Getting the mapping from class index to class label
  idx2label = dict((v,k) for k,v in label2index.items())

  predictions = model.predict_classes(validation_features)
  prob = model.predict(validation_features)

  errors = np.where(predictions != ground_truth)[0]
  print("No of errors = {}/{}".format(len(errors),nVal))
  # print(len(errors))
  for i in range(len(errors)):
      # print(i)
      pred_class = np.argmax(prob[errors[i]])
      # print(idx2label)
      # print(pred_class)
      pred_label = idx2label[0]
      # correc pred label
      # pred_label = idx2label[pred_class]

      print('Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
          fnames[errors[i]].split('/')[0],
          pred_label,
          prob[errors[i]][pred_class]))

      # original = load_img('{}/{}'.format(validation_dir,fnames[errors[i]]))
      # plt.imshow(original)
      # plt.show()

trainmethod(train_features, train_labels)

evalmethod(validation_features, validation_labels)

model()