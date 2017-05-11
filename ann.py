import numpy as np
import keras

#IMPORT PREPROCESSED DATA
from data_processing import data_preprocessing

X_train, X_test, X, y, y_train, y_test, sc = data_preprocessing()

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
model.fit(X_train, y_train, batch_size = 10, epochs = 10)

#TEST ACCURACY
test_accuracy = model.predict(X_test)
test_accuracy = (test_accuracy > 0.5)

#CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, test_accuracy)

#TESTING ON SPECIFIC CUSTOMER, DATA STRUCTURE: [germany, spain, credit_score, gender, age, tenure, balance, products, credit_card, active_member, salary]
customer_data = np.asarray([0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]).reshape(1,11)
feature_scale_data = sc.transform(customer_data)
customer_leaves = model.predict(feature_scale_data)