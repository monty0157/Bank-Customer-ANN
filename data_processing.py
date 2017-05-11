#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import pandas as pd

def data_preprocessing():
    #IMPORT DATA
    dataset = pd.read_csv(filepath_or_buffer = "~/documents/projects/deep_learning_course_udemy/bank_customer_ann/Data.csv")
    X = dataset.iloc[:, 3:13].values
    y = dataset.iloc[:, 13].values
    
    #ENCODE DATA
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelEncoder = LabelEncoder()
    
    X_encode_countries = labelEncoder.fit_transform(X[:, 1])
    X[:, 1] = X_encode_countries
    
    X_encode_gender = labelEncoder.fit_transform(X[:, 2])
    X[:, 2] = X_encode_gender
    
    #SPLIT COUNTRIES AND REMOVE DUMMY VARIABLE TRAP
    oneHotEncoder = OneHotEncoder(categorical_features = [1])
    X = oneHotEncoder.fit_transform(X).toarray()
    X = X[:, 1:12]
    
    #SPLIT DATA IN TEST AND TRAIN
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
    
    #FEATURE SCALING
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    return X_train, X_test, X, y, y_train, y_test, sc

