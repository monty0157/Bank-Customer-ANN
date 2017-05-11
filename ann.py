import numpy as np
import keras

#IMPORT PREPROCESSED DATA
from data_processing import data_preprocessing

X_train, X_test, X, y, y_train, y_test = data_preprocessing()

#BUILDING ANN MODEL
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

#TWO HIDDEN LAYERS AND OUTPUT LAYER
model.add(Dense(units = 6, activation = 'relu', kernel_initializer='uniform', input_shape = (11,)))
model.add(Dense(units = 6, activation = 'relu', kernel_initializer='uniform'))
model.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer='uniform'))

#GRADIENT DESCENT
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#DEPLOY ANN ON TRAINING SET
model.fit(X_train, y_train, batch_size = 10, epochs = 100)