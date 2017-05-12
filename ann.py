import numpy as np
import keras

#IMPORT PREPROCESSED DATA
from data_processing import data_preprocessing

X_train, X_test, X, y, y_train, y_test, sc = data_preprocessing()

#BUILDING ANN MODEL
from keras.models import Sequential
from keras.layers import Dense, Dropout

def build_model(optimizer = 'adam', units = 6):
    model = Sequential()

    #TWO HIDDEN LAYERS AND OUTPUT LAYER WITH DROPOUT
    model.add(Dense(units = units, activation = 'relu', kernel_initializer='uniform', input_shape = (11,)))
    model.add(Dropout(rate = 0.1))
    model.add(Dense(units = units, activation = 'relu', kernel_initializer='uniform'))
    model.add(Dropout(rate = 0.1))
    model.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer='uniform'))
    
    #GRADIENT DESCENT
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model

#DEPLOY ANN ON TRAINING SET WITH K-CROSS-VALIDATION
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

model = KerasClassifier(build_fn = build_model, batch_size = 10, nb_epoch = 10)
accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10, n_jobs = 1)
mean = accuracies.mean()
variance = accuracies.std()

#GRIDSEARCH
from sklearn.model_selection import GridSearchCV
grid_search = KerasClassifier(build_fn = build_model)
parameters = {
        'batch_size': [25,32],
        'nb_epoch': [10, 30],
        'optimizer': ['adam', 'rmsprop'],
        'units': [6, 10, 25],
        }
grid_search = GridSearchCV(estimator = model, param_grid = parameters, scoring = 'accuracy', cv = 10)
grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

##TEST ACCURACY
test_accuracy = model.predict(X_test)
test_accuracy = (test_accuracy > 0.5)

#CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, test_accuracy)

#TESTING ON SPECIFIC CUSTOMER, DATA STRUCTURE: [germany, spain, credit_score, gender, age, tenure, balance, products, credit_card, active_member, salary]
customer_data = np.asarray([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])
feature_scale_data = sc.transform(customer_data)
customer_leaves = model.predict(feature_scale_data)