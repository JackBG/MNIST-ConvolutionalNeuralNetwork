import tensorflow as tf
import pandas as pd
from keras.utils.np_utils import to_categorical
import numpy as np


# Loads .csv file or the keras dataset

# Import csv data
data = pd.read_csv('train.csv')
targets = data["label"]
targets = np.asarray(targets).astype('float32').reshape((-1,1))
features = data.drop(labels=["label"], axis=1)
features = features / 255.0
features = features.values.reshape(-1, 28, 28)

# Import csv test data
data = pd.read_csv('test.csv')
targets_test = data["label"]
targets_test = np.asarray(targets_test).astype('float32').reshape((-1, 1))
features_test = data.drop(labels=["label"], axis=1)
features_test = features_test / 255.0
features_test.values.reshape(-1, 28, 28)

# Import data from keras
data = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = data.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

# Model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)

model.fit(features, targets, epochs=5)
model.evaluate(features_test, targets_test)