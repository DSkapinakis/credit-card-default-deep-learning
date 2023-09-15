
######################### Importing Libraries - Setting seed ##################################

#Data Manipulation libraries
import pandas as pd 
import numpy as np
import copy

#Preprocessing - Cross valdidation - Evaluation metrics
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#Tensorflow/Keras library - Deep Learning models
import tensorflow
from keras.layers import SimpleRNN, LSTM, Conv1D, MaxPooling1D, Flatten, Concatenate, Dense
from keras.optimizers import Adam, RMSprop
from keras.callbacks import LearningRateScheduler
from keras import Input, Model, Sequential
import matplotlib.pyplot as plt

#Machine Learning models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

#Setting seed libraries
import os
import random
from tensorflow.random import set_seed
from keras.utils import set_random_seed
from keras import backend as K

# Setting seed value to 42
seed_value= 42
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tensorflow.random.set_seed(seed_value)
session_conf = tensorflow.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tensorflow.compat.v1.Session(graph=tensorflow.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)

######################### Data importing - Initial preprocessing ##################################

#Importing data
data = pd.read_excel('../data/default_of_credit_card_clients.xls',header=1)
data.head()

'''
Function for initial preprocessing of the data:
1) replace EDUCATION values 0, 5, 6 with 4 ('other' category) since they are not mentioned in the data description
2) replace MARRIAGE value 0 with 3 ('other' category) as there is not a 0 category for marriage column on data description 
3) drop 'ID' column - useless
4) rename target column to DEFAULT, rename PAY_0 to PAY_1 for consistency and more accurate variable names
'''
def initial_preprocessing(df):
    print('EDUCATION values before preprocessing:\n',df['EDUCATION'].value_counts())
    df['EDUCATION'].replace([0,5,6],4,inplace=True)
    print()
    print('EDUCATION values after preprocessing:\n',df['EDUCATION'].value_counts())
    print()
    print('MARRIAGE values before preprocessing:\n',df['MARRIAGE'].value_counts())
    df['MARRIAGE'].replace(0,3,inplace=True)
    print()
    print('MARRIAGE values after preprocessing:\n',df['MARRIAGE'].value_counts())
    df.drop(columns='ID',inplace=True)
    df.rename(columns={"default payment next month": "DEFAULT","PAY_0": "PAY_1"},inplace=True)
    
    return df

#perform initial preprocessing
data = initial_preprocessing(data)

######################### Modelling Pipeline ##################################

'''
- train test split (20% test) before scaling and encoding to prevent data leakages 
- train set will be used for cross-validation/hyperparameter tuning and test set for final evaluation 
- Stratify is used to ensure that the proportion of the class labels will remain consistent
'''
X_train, X_test, y_train, y_test = train_test_split(data.drop('DEFAULT',axis=1),data['DEFAULT'],
                                                    test_size=0.2,stratify=data['DEFAULT'],random_state=42)
print('X_train shape:',X_train.shape)
print('X_test shape:',X_test.shape)

## Column Transformer
#Column transformer for Robust Scaler and One Hot Encoder 
#making sure to return the preprocessed dataframes for better inspection
class PreprocessorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns,columns_num, drop='first', handle_unknown='ignore',sparse_output=False):
        self.columns = columns
        self.columns_num = columns_num
        self.drop = drop
        self.handle_unknown = handle_unknown
        self.sparse_output = sparse_output
        self.encoders = {}
        self.robust_enc = {}

    def fit(self, X, y=None):
        for col in self.columns:
            encoder = OneHotEncoder(drop=self.drop, sparse_output=self.sparse_output, 
                                    handle_unknown=self.handle_unknown)
            encoder.fit(X[[col]])
            self.encoders[col] = encoder
        
        for col_num in self.columns_num:
            encoder_robust = RobustScaler()
            encoder_robust.fit(X[[col_num]])
            self.robust_enc[col_num] = encoder_robust
            
        return self

    def transform(self, X):
        transformed = X.copy()
        for col in self.columns:
            encoder = self.encoders[col]
            encoded_cols = encoder.transform(transformed[[col]])
            new_cols = [f"{col}_{value}" for value in encoder.categories_[0][1:]]
            encoded_cols_df = pd.DataFrame(encoded_cols, columns=new_cols, index=transformed.index)
            transformed = pd.concat([transformed, encoded_cols_df], axis=1)
        transformed = transformed.drop(self.columns, axis=1)
        
        for col_num in self.columns_num:
            encoder_robust = self.robust_enc[col_num]
            transformed[col_num] = encoder_robust.transform(transformed[[col_num]])
            
        return transformed
    
## Preprocessing Pipeline
def preprocess_data(X_train, y_train, X_test, y_test):
    
    cat_cols = ['EDUCATION','MARRIAGE']
    numerical_cols = ['LIMIT_BAL', 'AGE','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5',
                  'BILL_AMT6','PAY_AMT1','PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    
    temp_cols = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5',
    'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4',
    'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
    'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    
    PAY_cols = ['PAY_6', 'PAY_5', 'PAY_4', 'PAY_3', 'PAY_2','PAY_1']
    BILL_AMT_cols = ['BILL_AMT6','BILL_AMT5','BILL_AMT4','BILL_AMT3','BILL_AMT2','BILL_AMT1']
    PAY_AMT_cols = ['PAY_AMT6','PAY_AMT5', 'PAY_AMT4', 'PAY_AMT3', 'PAY_AMT2', 'PAY_AMT1']
    
    #fitting the column transformer
    enc = PreprocessorTransformer(columns = cat_cols, columns_num= numerical_cols,
                                  drop='first',handle_unknown='ignore',sparse_output=False) 
    
    X_train_preprocessed = enc.fit_transform(X_train)
    X_test_preprocessed = enc.transform(X_test)
    
    static_cols_train = X_train_preprocessed.drop(temp_cols,axis=1).columns.to_list()
    static_cols_test = X_test_preprocessed.drop(temp_cols,axis=1).columns.to_list()
    
    #separation of static and temporal features
    X_train_temp = X_train_preprocessed[temp_cols]
    X_train_static = X_train_preprocessed[static_cols_train]
    X_test_temp = X_test_preprocessed[temp_cols]
    X_test_static = X_test_preprocessed[static_cols_test]

    PAY_train = X_train_temp[PAY_cols].to_numpy()
    BILL_AMT_train = X_train_temp[BILL_AMT_cols].to_numpy()
    PAY_AMT_train = X_train_temp[PAY_AMT_cols].to_numpy()
    
    PAY_test = X_test_temp[PAY_cols].to_numpy()
    BILL_AMT_test = X_test_temp[BILL_AMT_cols].to_numpy()
    PAY_AMT_test = X_test_temp[PAY_AMT_cols].to_numpy()   
    
    # Stacking temporal features
    stacked_train = np.dstack((PAY_train, BILL_AMT_train, PAY_AMT_train))
    stacked_test = np.dstack((PAY_test, BILL_AMT_test, PAY_AMT_test))
    y_train_preprocessed = y_train.to_numpy()
    y_test_preprocessed = y_test.to_numpy()
    
    return stacked_train, X_train_static, y_train_preprocessed, stacked_test, X_test_static, y_test_preprocessed

######################### Hyperparameter Tuning ##################################

# 5-fold cross-validation
n_splits = 5 
kf = StratifiedKFold(n_splits=n_splits)

# Function to receive temporal features and make predictions with the ML model as FCL
def classifier_prediction_temporal(X_train_temp, X_test_temp, y_train_prep,
                                   model, feature_extractor_model, layer_name):
    
    # Extract features using the feature_extractor_model
    extractor_model = Model(inputs=feature_extractor_model.input, 
                            outputs=feature_extractor_model.get_layer(name=layer_name).output)
    customers_vector = extractor_model.predict(X_train_temp)
    customers_test = extractor_model.predict(X_test_temp)
    
    # Reshape the extracted features
    reshaped_customers_vector = customers_vector.reshape(customers_vector.shape[0], -1)
    reshaped_customers_test = customers_test.reshape(customers_test.shape[0], -1) 
    
    # Train the classification model with the extracted features
    final_model = model
    final_model.fit(reshaped_customers_vector, y_train_prep)
    
    # Make predictions using the trained model
    preds = final_model.predict(reshaped_customers_test)
    preds_train = final_model.predict(reshaped_customers_vector)
    
    return preds, preds_train

# Function to receive both static and temporal features and make predictions with the ML model as FCL
def classifier_prediction(X_train_temp,X_test_temp,X_train_st,
                          X_test_st,y_train_prep,y_test_prep,model, feature_extractor_model, layer_name):
    
    extractor_model = Model(inputs=feature_extractor_model.input, 
                            outputs=feature_extractor_model.get_layer(name=layer_name).output)
    customers_vector = extractor_model.predict([X_train_temp,X_train_st])
    customers_test = extractor_model.predict([X_test_temp,X_test_st])
    reshaped_customers_vector = customers_vector.reshape(customers_vector.shape[0], -1)
    reshaped_customers_test = customers_test.reshape(customers_test.shape[0], -1) 
    final_model = model
    final_model.fit(reshaped_customers_vector, y_train_prep)
    preds = final_model.predict(reshaped_customers_test)
    preds_train = final_model.predict(reshaped_customers_vector)
    
    return preds, preds_train

## Architecture Tuning
### Temporal features

#### RNN Temporal (Dense)
#insert the respective architecture configurations to be tested
architectures = [
    {'hidden_layers': 4, 'units_per_layer': 32},
    {'hidden_layers': 3, 'units_per_layer': 128}    
]

best_architecture = None
best_f1_score = 0.0

for architecture in architectures:
    fold_no = 1
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for train_index, test_index in kf.split(X_train, y_train):
        
        X_train_dl, X_test_dl = X_train.iloc[train_index,:], X_train.iloc[test_index,:]
        y_train_dl, y_test_dl = y_train.iloc[train_index], y_train.iloc[test_index]

        stacked_train, X_train_static, y_train_dl, stacked_test,\
            X_test_static, y_test_dl = preprocess_data(X_train_dl,y_train_dl, X_test_dl, y_test_dl)
        
        num_time_steps, num_features = stacked_train.shape[1], stacked_train.shape[2]
        
        # Build the RNN model
        model = Sequential()
        model.add(SimpleRNN(architecture['units_per_layer'], return_sequences=True, 
                            input_shape=(num_time_steps, num_features)))
        for _ in range(1, architecture['hidden_layers']):
            model.add(SimpleRNN(architecture['units_per_layer'], return_sequences=True))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        
        initial_learning_rate = 0.0001  # Initial learning rate
        decay_rate = 0.1  # Decay rate
        decay_steps = 20  # Decay steps (number of steps before applying decay)
        epochs = 50

        def learning_rate_scheduler(epoch):
            return initial_learning_rate * decay_rate ** (epoch // decay_steps)
        
        
        model.compile(loss='binary_crossentropy', 
                      optimizer=Adam(learning_rate=initial_learning_rate), metrics=['accuracy'])
        
        lr_scheduler = LearningRateScheduler(learning_rate_scheduler)
        
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no}')
    
        batch_size = 64
        
        # Train the model
        history = model.fit(stacked_train, y_train_dl, epochs=epochs, batch_size=batch_size,
                            validation_data=(stacked_test, y_test_dl),
                            shuffle=False,callbacks=[lr_scheduler])
        
        # Plot the loss on train vs validate tests
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        
        y_pred_probs = model.predict(stacked_test)
        y_pred = (y_pred_probs>=0.5).astype(int)    
        
        accuracy =  accuracy_score(y_test_dl,y_pred)
        precision = precision_score(y_test_dl,y_pred)
        recall = recall_score(y_test_dl,y_pred)
        f1 =  f1_score(y_test_dl,y_pred)

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        
        fold_no = fold_no + 1
        
    # Calculate the average F1 score for the current architecture
    average_f1_score = np.mean(f1_scores)

    # Print the scores for the current architecture
    print(f"Architecture: {architecture}")
    print('Accuracy: %.3f (+/- %.3f)' % (np.mean(accuracy_scores), np.std(accuracy_scores)))
    print('Precision: %.3f (+/- %.3f)' % (np.mean(precision_scores), np.std(precision_scores)))
    print('Recall: %.3f (+/- %.3f)' % (np.mean(recall_scores), np.std(recall_scores)))
    print('F1 score: %.3f (+/- %.3f)' % (np.mean(f1_scores), np.std(f1_scores)))
    print()
    
    # Check if the current architecture has a higher average F1 score
    # If so, update the best architecture and best F1 score
    if average_f1_score > best_f1_score:
        best_architecture = architecture
        best_f1_score = average_f1_score


# Print the best architecture
print("Best Architecture:")
print(best_architecture)
print("Best F1 Score:", best_f1_score)

#### LSTM Temporal (Dense)
#insert the respective architecture configurations to be tested
architectures = [
    {'hidden_layers': 5, 'units_per_layer': 32},
    {'hidden_layers': 6, 'units_per_layer': 32}
]

best_architecture = None
best_f1_score = 0.0

for architecture in architectures:
    fold_no = 1
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for train_index, test_index in kf.split(X_train, y_train):
        
        X_train_dl, X_test_dl = X_train.iloc[train_index,:], X_train.iloc[test_index,:]
        y_train_dl, y_test_dl = y_train.iloc[train_index], y_train.iloc[test_index]

        stacked_train, X_train_static, y_train_dl, stacked_test, X_test_static,
        y_test_dl = preprocess_data(X_train_dl,y_train_dl, X_test_dl, y_test_dl)
        
        num_time_steps, num_features = stacked_train.shape[1], stacked_train.shape[2]
        
        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(architecture['units_per_layer'], return_sequences=True, 
                       input_shape=(num_time_steps, num_features)))
        for _ in range(1, architecture['hidden_layers']):
            model.add(LSTM(architecture['units_per_layer'], return_sequences=True))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        
        initial_learning_rate = 0.0001  # Initial learning rate
        decay_rate = 0.1  # Decay rate
        decay_steps = 20  # Decay steps (number of steps before applying decay)
        epochs = 50

        def learning_rate_scheduler(epoch):
            return initial_learning_rate * decay_rate ** (epoch // decay_steps)
        
        
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=initial_learning_rate),
                      metrics=['accuracy'])
        
        lr_scheduler = LearningRateScheduler(learning_rate_scheduler)
        
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no}')
    
        batch_size = 64
        
        # Train the model
        history = model.fit(stacked_train, y_train_dl, epochs=epochs, batch_size=batch_size, 
                            validation_data=(stacked_test, y_test_dl),
                            shuffle=False,callbacks=[lr_scheduler])
        
        # Plot the loss on train vs validate tests
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        
        y_pred_probs = model.predict(stacked_test)
        y_pred = (y_pred_probs>=0.5).astype(int)    
        
        accuracy =  accuracy_score(y_test_dl,y_pred)
        precision = precision_score(y_test_dl,y_pred)
        recall = recall_score(y_test_dl,y_pred)
        f1 =  f1_score(y_test_dl,y_pred)

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        
        fold_no = fold_no + 1
        
    # Calculate the average F1 score for the current architecture
    average_f1_score = np.mean(f1_scores)
    total_params = model.count_params()  

    # Print the scores for the current architecture
    print(f"Architecture: {architecture}")
    print('Accuracy: %.3f (+/- %.3f)' % (np.mean(accuracy_scores), np.std(accuracy_scores)))
    print('Precision: %.3f (+/- %.3f)' % (np.mean(precision_scores), np.std(precision_scores)))
    print('Recall: %.3f (+/- %.3f)' % (np.mean(recall_scores), np.std(recall_scores)))
    print('F1 score: %.3f (+/- %.3f)' % (np.mean(f1_scores), np.std(f1_scores)))
    print('total params:', total_params)
    print()
    
    # Check if the current architecture has a higher average F1 score
    # If so, update the best architecture and best F1 score
    if average_f1_score > best_f1_score:
        best_architecture = architecture
        best_f1_score = average_f1_score

# Print the best architecture
print("Best Architecture:")
print(best_architecture)
print("Best F1 Score:", best_f1_score)

#### CNN Temporal (Dense)
#insert the respective architecture configurations to be tested
first_layer_filters = [128,256]

best_configuration = None
best_f1_score = 0.0

for filter in first_layer_filters:
            
    fold_no = 1
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for train_index, test_index in kf.split(X_train, y_train):
        
        X_train_dl, X_test_dl = X_train.iloc[train_index,:], X_train.iloc[test_index,:]
        y_train_dl, y_test_dl = y_train.iloc[train_index], y_train.iloc[test_index]

        stacked_train, X_train_static, y_train_dl, stacked_test, 
        X_test_static, y_test_dl = preprocess_data(X_train_dl,y_train_dl, X_test_dl, y_test_dl)
        
        num_time_steps, num_features = stacked_train.shape[1], stacked_train.shape[2]
        
        # Build the CNN model
        model = Sequential()
        model.add(Conv1D(filters=filter, kernel_size=5, activation='relu',padding='same',
                         input_shape=(num_time_steps, num_features)))
        model.add(MaxPooling1D(pool_size=3))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        
        initial_learning_rate = 0.001  # Initial learning rate
        decay_rate = 0.1  # Decay rate
        decay_steps = 20  # Decay steps (number of steps before applying decay)
        epochs = 50

        def learning_rate_scheduler(epoch):
            return initial_learning_rate * decay_rate ** (epoch // decay_steps)
        
        model.compile(loss='binary_crossentropy', 
                      optimizer=Adam(learning_rate = initial_learning_rate), metrics=['accuracy'])
        
        lr_scheduler = LearningRateScheduler(learning_rate_scheduler)
        
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no}')
        
        batch_size = 64
        
        # Train the model
        history = model.fit(stacked_train, y_train_dl, epochs=epochs, batch_size=batch_size,
                            validation_data=(stacked_test, y_test_dl),
                            shuffle=False,callbacks=[lr_scheduler])
        
        # Plot the loss on train vs validate tests
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        
        y_pred_probs = model.predict(stacked_test)
        y_pred = (y_pred_probs>=0.5).astype(int)    
        
        accuracy =  accuracy_score(y_test_dl,y_pred)
        precision = precision_score(y_test_dl,y_pred)
        recall = recall_score(y_test_dl,y_pred)
        f1 =  f1_score(y_test_dl,y_pred)

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        
        fold_no = fold_no + 1
    
    # Calculate the average F1 score for the current configuration
    average_f1_score = np.mean(f1_scores)
    total_params = model.count_params()

    # Print the scores for the current configuration
    print(f"Configuration: filters=({filter})")
    print('Accuracy: %.3f (+/- %.3f)' % (np.mean(accuracy_scores), np.std(accuracy_scores)))
    print('Precision: %.3f (+/- %.3f)' % (np.mean(precision_scores), np.std(precision_scores)))
    print('Recall: %.3f (+/- %.3f)' % (np.mean(recall_scores), np.std(recall_scores)))
    print('F1 score: %.3f (+/- %.3f)' % (np.mean(f1_scores), np.std(f1_scores)))
    print("Total Trainable Parameters of CNN:", total_params)
    print()

    # Check if the current configuration has a higher average F1 score
    # If so, update the best configuration and best F1 score
    if average_f1_score > best_f1_score:
        best_configuration = (filter)
        best_f1_score = average_f1_score


# Print the best configuration
print("Best Configuration of filters:")
print(best_configuration)
print("Best F1 Score:", best_f1_score)

#### RNN Temporal + ML Models
def grid_rnn_temporal(class_model):
   
#insert the respective architecture configurations to be tested   
    architectures = [
        {'hidden_layers': 4, 'units_per_layer': 32},
        {'hidden_layers': 3, 'units_per_layer': 32},
        {'hidden_layers': 4, 'units_per_layer': 16}
        
        ]

    best_architecture = None
    best_f1_score = 0.0

    for architecture in architectures:
        fold_no = 1
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for train_index, test_index in kf.split(X_train, y_train):
            
            X_train_dl, X_test_dl = X_train.iloc[train_index,:], X_train.iloc[test_index,:]
            y_train_dl, y_test_dl = y_train.iloc[train_index], y_train.iloc[test_index]

            stacked_train, X_train_static, y_train_dl, 
            stacked_test, X_test_static, y_test_dl = preprocess_data(X_train_dl,y_train_dl, 
                                                                     X_test_dl, y_test_dl)
        
            num_time_steps, num_features = stacked_train.shape[1], stacked_train.shape[2]
            
            # Build the RNN model
            model = Sequential()
            model.add(SimpleRNN(architecture['units_per_layer'],
                                return_sequences=True, input_shape=(num_time_steps, num_features)))
            for _ in range(1, architecture['hidden_layers']):
                model.add(SimpleRNN(architecture['units_per_layer'], return_sequences=True))
            model.add(Flatten(name='RNN_FLATTEN'))
            model.add(Dense(1, activation='sigmoid'))
            
            initial_learning_rate = 0.0001  # Initial learning rate
            decay_rate = 0.1  # Decay rate
            decay_steps = 20  # Decay steps (number of steps before applying decay)
            batch_size = 64
            epochs = 50

            def learning_rate_scheduler(epoch):
                return initial_learning_rate * decay_rate ** (epoch // decay_steps)
            
            
            model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=initial_learning_rate), 
                          metrics=['accuracy'])
            
            print('------------------------------------------------------------------------')
            print(f'Training for fold {fold_no}')
        
            lr_scheduler = LearningRateScheduler(learning_rate_scheduler)
        
            # Train the model
            history = model.fit(stacked_train, y_train_dl, epochs=epochs, batch_size=batch_size,
                                validation_data=(stacked_test, y_test_dl),
                                shuffle=False,callbacks=[lr_scheduler])
            
            # Plot the loss on train vs validate tests
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
            
            preds = classifier_prediction_temporal(stacked_train,stacked_test, y_train_dl,
                                                   model = class_model, feature_extractor_model=model, 
                                                   layer_name='RNN_FLATTEN')[0]
            
            accuracy =  accuracy_score(y_test_dl,preds)
            precision = precision_score(y_test_dl,preds)
            recall = recall_score(y_test_dl,preds)
            f1 =  f1_score(y_test_dl,preds)

            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            
            fold_no = fold_no + 1
          
        # Calculate the number of parameters for the current architecture
        total_params = model.count_params()  
            
        # Calculate the average F1 score for the current architecture
        average_f1_score = np.mean(f1_scores)

        # Print the scores for the current architecture
        print(f"Architecture: {architecture}")
        print('Accuracy: %.3f (+/- %.3f)' % (np.mean(accuracy_scores), np.std(accuracy_scores)))
        print('Precision: %.3f (+/- %.3f)' % (np.mean(precision_scores), np.std(precision_scores)))
        print('Recall: %.3f (+/- %.3f)' % (np.mean(recall_scores), np.std(recall_scores)))
        print('F1 score: %.3f (+/- %.3f)' % (np.mean(f1_scores), np.std(f1_scores)))
        print("Total Trainable Parameters of RNN:", total_params)
        print()
        
        # Check if the current architecture has a higher average F1 score
        # If so, update the best architecture and best F1 score
        if average_f1_score > best_f1_score:
            best_architecture = architecture
            best_f1_score = average_f1_score


    # Print the best architecture
    print("Best Architecture:")
    print(best_architecture)
    print("Best F1 Score:", best_f1_score)
    
grid_rnn_temporal(LogisticRegression())
grid_rnn_temporal(RandomForestClassifier())
grid_rnn_temporal(XGBClassifier())

#### LSTM Temporal + ML Models
def grid_lstm_temporal(class_model):
   
   #insert the respective architecture configurations to be tested
   
    architectures = [
        {'hidden_layers': 1, 'units_per_layer': 128},
        {'hidden_layers': 2, 'units_per_layer': 64},
        {'hidden_layers': 3, 'units_per_layer': 64},
        {'hidden_layers': 2, 'units_per_layer': 128},
        {'hidden_layers': 2, 'units_per_layer': 256},
        {'hidden_layers': 3, 'units_per_layer': 128},
        {'hidden_layers': 4, 'units_per_layer': 16},
        {'hidden_layers': 3, 'units_per_layer': 32},
        {'hidden_layers': 4, 'units_per_layer': 32},
        {'hidden_layers': 4, 'units_per_layer': 64},
        {'hidden_layers': 5, 'units_per_layer': 32},
        {'hidden_layers': 6, 'units_per_layer': 32}
    ]

    best_architecture = None
    best_f1_score = 0.0

    for architecture in architectures:
        fold_no = 1
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for train_index, test_index in kf.split(X_train, y_train):
            
            X_train_dl, X_test_dl = X_train.iloc[train_index,:], X_train.iloc[test_index,:]
            y_train_dl, y_test_dl = y_train.iloc[train_index], y_train.iloc[test_index]

            stacked_train, X_train_static, y_train_dl, stacked_test, 
            X_test_static, y_test_dl = preprocess_data(X_train_dl,y_train_dl, X_test_dl, y_test_dl)
            
            num_time_steps, num_features = stacked_train.shape[1], stacked_train.shape[2]
            
            # Build the LSTM model
            model = Sequential()
            model.add(LSTM(architecture['units_per_layer'],
                           return_sequences=True, input_shape=(num_time_steps, num_features)))
            for _ in range(1, architecture['hidden_layers']):
                model.add(LSTM(architecture['units_per_layer'], return_sequences=True))
            model.add(Flatten(name='LSTM_FLATTEN'))
            model.add(Dense(1, activation='sigmoid'))
            
            initial_learning_rate = 0.0001  # Initial learning rate
            decay_rate = 0.1  # Decay rate
            decay_steps = 20  # Decay steps (number of steps before applying decay)
            batch_size = 64
            epochs = 50

            def learning_rate_scheduler(epoch):
                return initial_learning_rate * decay_rate ** (epoch // decay_steps)
            
            
            model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=initial_learning_rate), 
                          metrics=['accuracy'])
            total_params = model.count_params()
            print('------------------------------------------------------------------------')
            print(f'Training for fold {fold_no}')
        
            lr_scheduler = LearningRateScheduler(learning_rate_scheduler)
        
            # Train the model
            history = model.fit(stacked_train, y_train_dl, epochs=epochs, batch_size=batch_size,
                                validation_data=(stacked_test, y_test_dl),
                                shuffle=False,callbacks=[lr_scheduler])
            
            # Plot the loss on train vs validate tests
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
            
            preds = classifier_prediction_temporal(stacked_train,stacked_test, y_train_dl,
                                                   model = class_model, feature_extractor_model=model, 
                                                   layer_name='LSTM_FLATTEN')[0]
            
            accuracy =  accuracy_score(y_test_dl,preds)
            precision = precision_score(y_test_dl,preds)
            recall = recall_score(y_test_dl,preds)
            f1 =  f1_score(y_test_dl,preds)

            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            
            fold_no = fold_no + 1
            
        # Calculate the average F1 score for the current architecture
        average_f1_score = np.mean(f1_scores)

        # Print the scores for the current architecture
        print(f"Architecture: {architecture}")
        print('Accuracy: %.3f (+/- %.3f)' % (np.mean(accuracy_scores), np.std(accuracy_scores)))
        print('Precision: %.3f (+/- %.3f)' % (np.mean(precision_scores), np.std(precision_scores)))
        print('Recall: %.3f (+/- %.3f)' % (np.mean(recall_scores), np.std(recall_scores)))
        print('F1 score: %.3f (+/- %.3f)' % (np.mean(f1_scores), np.std(f1_scores)))
        print("Total Trainable Parameters:", total_params)
        print()
        
        # Check if the current architecture has a higher average F1 score
        # If so, update the best architecture and best F1 score
        if average_f1_score > best_f1_score:
            best_architecture = architecture
            best_f1_score = average_f1_score

    # Print the best architecture
    print("Best Architecture:")
    print(best_architecture)
    print("Best F1 Score:", best_f1_score)
    
grid_lstm_temporal(LogisticRegression())
grid_lstm_temporal(RandomForestClassifier())
grid_lstm_temporal(XGBClassifier())

#### CNN Temporal + ML Models
def grid_cnn_temporal(class_model):
    
#insert the respective architecture configurations to be tested
    first_layer_filters = [64,128,256]

    best_configuration = None
    best_f1_score = 0.0

    for filter in first_layer_filters:
        
                
        fold_no = 1
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for train_index, test_index in kf.split(X_train, y_train):
            
            X_train_dl, X_test_dl = X_train.iloc[train_index,:], X_train.iloc[test_index,:]
            y_train_dl, y_test_dl = y_train.iloc[train_index], y_train.iloc[test_index]

            stacked_train, X_train_static, y_train_dl, stacked_test,
            X_test_static, y_test_dl = preprocess_data(X_train_dl,y_train_dl, X_test_dl, y_test_dl)
            
            num_time_steps, num_features = stacked_train.shape[1], stacked_train.shape[2]
            
            # Build the CNN model
            model = Sequential()
            model.add(Conv1D(filters=filter, kernel_size=5, activation='relu', 
                             padding='same', input_shape=(num_time_steps, num_features)))
            model.add(MaxPooling1D(pool_size=3))
            model.add(Flatten(name='CNN_FLATTEN'))
            model.add(Dense(1, activation='sigmoid'))
            
            initial_learning_rate = 0.001  # Initial learning rate
            decay_rate = 0.1  # Decay rate
            decay_steps = 20  # Decay steps (number of steps before applying decay)
            batch_size = 64
            epochs = 50

            def learning_rate_scheduler(epoch):
                return initial_learning_rate * decay_rate ** (epoch // decay_steps)
            
            model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate = initial_learning_rate), 
                          metrics=['accuracy'])
            
            print('------------------------------------------------------------------------')
            print(f'Training for fold {fold_no}')
            
            lr_scheduler = LearningRateScheduler(learning_rate_scheduler)
            
            # Train the model
            history = model.fit(stacked_train, y_train_dl, epochs=epochs, batch_size=batch_size, 
                                validation_data=(stacked_test, y_test_dl),shuffle=False,callbacks=[lr_scheduler])
            
            # Plot the loss on train vs validate tests
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
            
            preds = classifier_prediction_temporal(stacked_train,stacked_test, y_train_dl, 
                                                   model = class_model, feature_extractor_model=model, 
                                                   layer_name='CNN_FLATTEN')[0]
            
            accuracy =  accuracy_score(y_test_dl,preds)
            precision = precision_score(y_test_dl,preds)
            recall = recall_score(y_test_dl,preds)
            f1 =  f1_score(y_test_dl,preds)

            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            
            fold_no = fold_no + 1
            
        # Calculate the number of parameters for the current configuration
        total_params = model.count_params()
        # Calculate the average F1 score for the current configuration
        average_f1_score = np.mean(f1_scores)

        # Print the scores for the current configuration
        print(f"Configuration: filters=({filter})")
        print('Accuracy: %.3f (+/- %.3f)' % (np.mean(accuracy_scores), np.std(accuracy_scores)))
        print('Precision: %.3f (+/- %.3f)' % (np.mean(precision_scores), np.std(precision_scores)))
        print('Recall: %.3f (+/- %.3f)' % (np.mean(recall_scores), np.std(recall_scores)))
        print('F1 score: %.3f (+/- %.3f)' % (np.mean(f1_scores), np.std(f1_scores)))
        print("Total Trainable Parameters:", total_params)
        print()

        # Check if the current configuration has a higher average F1 score
        # If so, update the best configuration and best F1 score
        if average_f1_score > best_f1_score:
            best_configuration = (filter)
            best_f1_score = average_f1_score


    # Print the best configuration
    print("Best Configuration of filters:")
    print(best_configuration)
    print("Best F1 Score:", best_f1_score)
    
grid_cnn_temporal(LogisticRegression())
grid_cnn_temporal(RandomForestClassifier())
grid_cnn_temporal(XGBClassifier())

### Static + Temporal features
#### RNN Concat (Dense)
#insert the respective architecture configurations to be tested

architectures = [

    {'hidden_layers': 3, 'units_per_layer': 64},
    {'hidden_layers': 3, 'units_per_layer': 128},
    {'hidden_layers': 4, 'units_per_layer': 16},
    {'hidden_layers': 3, 'units_per_layer': 32},
    {'hidden_layers': 4, 'units_per_layer': 32},
    {'hidden_layers': 4, 'units_per_layer': 64},
    {'hidden_layers': 5, 'units_per_layer': 32},
    {'hidden_layers': 6, 'units_per_layer': 32},
]

best_architecture = None
best_f1_score = 0.0

for architecture in architectures:
    fold_no = 1
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for train_index, test_index in kf.split(X_train, y_train):
        
        X_train_dl, X_test_dl = X_train.iloc[train_index, :], X_train.iloc[test_index, :]
        y_train_dl, y_test_dl = y_train.iloc[train_index], y_train.iloc[test_index]

        stacked_train, X_train_static, y_train_dl, stacked_test, 
        X_test_static, y_test_dl = preprocess_data(X_train_dl, y_train_dl, X_test_dl, y_test_dl)
        
        # Input layers
        num_time_steps, num_features = stacked_train.shape[1], stacked_train.shape[2]

        temporal_input = Input(shape=(num_time_steps, num_features), name='TEMPORAL_INPUT')
        static_input = Input(shape=(X_train_static.shape[1]), name='STATIC_INPUT')

        # RNN layers
        rnn_layer = SimpleRNN(architecture['units_per_layer'], 
                              return_sequences=True, name=f'RNN_LAYER_1')(temporal_input)
        for i in range(1, architecture['hidden_layers']):
            rnn_layer = SimpleRNN(architecture['units_per_layer'],
                                  return_sequences=True, name=f'RNN_LAYER_{i + 1}')(rnn_layer)
        rnn_layer = Flatten(name='FLATTEN')(rnn_layer)

        # Concatenate RNN layer with static input
        rnn_combined = Concatenate(axis=1, name='rnn_CONCAT')([rnn_layer, static_input])
        output = Dense(1, activation='sigmoid', name='rnn_OUTPUT_LAYER')(rnn_combined)

        model = Model(inputs=[temporal_input, static_input], outputs=[output])

        initial_learning_rate = 0.0001
        decay_rate = 0.1
        decay_steps = 20
        batch_size = 64
        epochs = 50

        def learning_rate_scheduler(epoch):
            return initial_learning_rate * decay_rate ** (epoch // decay_steps)

        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(learning_rate=initial_learning_rate), metrics=['accuracy'])
        
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no}')

        lr_scheduler = LearningRateScheduler(learning_rate_scheduler)
        history = model.fit([stacked_train, X_train_static], y_train_dl,
                                                epochs=epochs, batch_size=batch_size,
                                                validation_data=([stacked_test, X_test_static], y_test_dl),
                                                shuffle=False, callbacks=[lr_scheduler])

        # Plot the loss on train vs validate tests
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Make predictions on the test set
        y_pred_probs = model.predict([stacked_test, X_test_static])
        y_pred = (y_pred_probs >= 0.5).astype(int)

        accuracy = accuracy_score(y_test_dl, y_pred)
        precision = precision_score(y_test_dl, y_pred)
        recall = recall_score(y_test_dl, y_pred)
        f1 = f1_score(y_test_dl, y_pred)

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        fold_no = fold_no + 1
        
    # Calculate the number of parameters for the current architecture
    total_params = model.count_params()  
    # Calculate the average F1 score for the current architecture
    average_f1_score = np.mean(f1_scores)

    # Print the scores for the current architecture
    print(f"Architecture: {architecture['hidden_layers']} RNN layers, {architecture['units_per_layer']} units")
    print('Accuracy: %.3f (+/- %.3f)' % (np.mean(accuracy_scores), np.std(accuracy_scores)))
    print('Precision: %.3f (+/- %.3f)' % (np.mean(precision_scores), np.std(precision_scores)))
    print('Recall: %.3f (+/- %.3f)' % (np.mean(recall_scores), np.std(recall_scores)))
    print('F1 score: %.3f (+/- %.3f)' % (np.mean(f1_scores), np.std(f1_scores)))
    print("Total Trainable Parameters:", total_params)
    print()

    # Check if the current architecture has a higher average F1 score
    # If so, update the best architecture and best F1 score
    if average_f1_score > best_f1_score:
        best_architecture = architecture
        best_f1_score = average_f1_score

# Print the best architecture
print("Best Architecture:")
print(best_architecture)
print("Best F1 Score:", best_f1_score)

#### LSTM Concat (Dense)
#insert the respective architecture configurations to be tested
architectures = [
    
    {'hidden_layers': 2, 'units_per_layer': 256}

]

best_architecture = None
best_f1_score = 0.0

for architecture in architectures:
    fold_no = 1
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for train_index, test_index in kf.split(X_train, y_train):
        
        X_train_dl, X_test_dl = X_train.iloc[train_index, :], X_train.iloc[test_index, :]
        y_train_dl, y_test_dl = y_train.iloc[train_index], y_train.iloc[test_index]

        stacked_train, X_train_static, y_train_dl, stacked_test,
        X_test_static, y_test_dl = preprocess_data(X_train_dl, y_train_dl, X_test_dl, y_test_dl)
        
        # Input layers
        num_time_steps, num_features = stacked_train.shape[1], stacked_train.shape[2]

        temporal_input = Input(shape=(num_time_steps, num_features), name='TEMPORAL_INPUT')
        static_input = Input(shape=(X_train_static.shape[1]), name='STATIC_INPUT')

        # LSTM layers
        lstm_layer = LSTM(architecture['units_per_layer'], 
                          return_sequences=True, name=f'LSTM_LAYER_1')(temporal_input)
        for i in range(1, architecture['hidden_layers']):
            lstm_layer = LSTM(architecture['units_per_layer'], 
                              return_sequences=True, name=f'LSTM_LAYER_{i + 1}')(lstm_layer)
        lstm_layer = Flatten(name='FLATTEN')(lstm_layer)

        # Concatenate LSTM layer with static input
        LSTM_combined = Concatenate(axis=1, name='LSTM_CONCAT')([lstm_layer, static_input])
        output = Dense(1, activation='sigmoid', name='LSTM_OUTPUT_LAYER')(LSTM_combined)

        model = Model(inputs=[temporal_input, static_input], outputs=[output])

        initial_learning_rate = 0.0001
        decay_rate = 0.1
        decay_steps = 20
        batch_size = 64
        epochs = 50

        def learning_rate_scheduler(epoch):
            return initial_learning_rate * decay_rate ** (epoch // decay_steps)

        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(learning_rate=initial_learning_rate), metrics=['accuracy'])
        
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no}')

        lr_scheduler = LearningRateScheduler(learning_rate_scheduler)
        history = model.fit([stacked_train, X_train_static], y_train_dl,
                                                epochs=epochs, batch_size=batch_size,
                                                validation_data=([stacked_test, X_test_static], y_test_dl),
                                                shuffle=False, callbacks=[lr_scheduler])

        # Plot the loss on train vs validate tests
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Make predictions on the test set
        y_pred_probs = model.predict([stacked_test, X_test_static])
        y_pred = (y_pred_probs >= 0.5).astype(int)

        accuracy = accuracy_score(y_test_dl, y_pred)
        precision = precision_score(y_test_dl, y_pred)
        recall = recall_score(y_test_dl, y_pred)
        f1 = f1_score(y_test_dl, y_pred)

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        fold_no = fold_no + 1
        
    # Calculate the number of parameters for the current architecture
    total_params = model.count_params()  
    # Calculate the average F1 score for the current architecture
    average_f1_score = np.mean(f1_scores)

    # Print the scores for the current architecture
    print(f"Architecture: {architecture['hidden_layers']} LSTM layers, {architecture['units_per_layer']} units")
    print('Accuracy: %.3f (+/- %.3f)' % (np.mean(accuracy_scores), np.std(accuracy_scores)))
    print('Precision: %.3f (+/- %.3f)' % (np.mean(precision_scores), np.std(precision_scores)))
    print('Recall: %.3f (+/- %.3f)' % (np.mean(recall_scores), np.std(recall_scores)))
    print('F1 score: %.3f (+/- %.3f)' % (np.mean(f1_scores), np.std(f1_scores)))
    print("Total Trainable Parameters:", total_params)
    print()

    # Check if the current architecture has a higher average F1 score
    # If so, update the best architecture and best F1 score
    if average_f1_score > best_f1_score:
        best_architecture = architecture
        best_f1_score = average_f1_score

# Print the best architecture
print("Best Architecture:")
print(best_architecture)
print("Best F1 Score:", best_f1_score)

#### CNN Concat (Dense)
#insert the respective architecture configurations to be tested
first_layer_filters = [4,8]

best_configuration = None
best_f1_score = 0.0

for filter in first_layer_filters:
      
    fold_no = 1
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for train_index, test_index in kf.split(X_train, y_train):
        
        X_train_dl, X_test_dl = X_train.iloc[train_index,:], X_train.iloc[test_index,:]
        y_train_dl, y_test_dl = y_train.iloc[train_index], y_train.iloc[test_index]

        stacked_train, X_train_static, y_train_dl, stacked_test, 
        X_test_static, y_test_dl = preprocess_data(X_train_dl,y_train_dl, X_test_dl, y_test_dl)

        num_time_steps, num_features = stacked_train.shape[1], stacked_train.shape[2]
        
        temporal_input = Input(shape=(num_time_steps, num_features), name='TEMPORAL_INPUT')
        static_input = Input(shape=(X_train_static.shape[1]), name='STATIC_INPUT')
        
        # CNN layers
        cnn_layer = Conv1D(filters=filter, kernel_size=5, activation='relu',padding='same',
                           name=f'CNN_LAYER_1')(temporal_input)
        cnn_layer = MaxPooling1D(pool_size=3)(cnn_layer)
        cnn_layer = Conv1D(filters=filter*2, kernel_size=3, activation='relu',padding='same',
                           name=f'CNN_LAYER_2')(cnn_layer)
        cnn_layer = MaxPooling1D(pool_size=2)(cnn_layer)
        cnn_layer = Flatten(name='FLATTEN')(cnn_layer)

        # Concatenate CNN layer with static input
        cnn_combined = Concatenate(axis=1, name='cnn_CONCAT')([cnn_layer, static_input])
        output = Dense(1, activation='sigmoid', name='cnn_OUTPUT_LAYER')(cnn_combined)
        
        model = Model(inputs=[temporal_input, static_input], outputs=[output])
        
        initial_learning_rate = 0.001  # Initial learning rate
        decay_rate = 0.1  # Decay rate
        decay_steps = 20  # Decay steps (number of steps before applying decay)
        batch_size = 64
        epochs = 50

        def learning_rate_scheduler(epoch):
            return initial_learning_rate * decay_rate ** (epoch // decay_steps)
        
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate = initial_learning_rate), 
                      metrics=['accuracy'])
        
        lr_scheduler = LearningRateScheduler(learning_rate_scheduler)
        
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no}')
    
        # Train the model
        history = model.fit([stacked_train, X_train_static], y_train_dl,
                                                epochs=epochs, batch_size=batch_size,
                                                validation_data=([stacked_test, X_test_static], y_test_dl),
                                                shuffle=False, callbacks=[lr_scheduler])
        
        # Plot the loss on train vs validate tests
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        
        # Make predictions on the test set
        y_pred_probs = model.predict([stacked_test, X_test_static])
        y_pred = (y_pred_probs >= 0.5).astype(int)

        accuracy = accuracy_score(y_test_dl, y_pred)
        precision = precision_score(y_test_dl, y_pred)
        recall = recall_score(y_test_dl, y_pred)
        f1 = f1_score(y_test_dl, y_pred)

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        
        fold_no = fold_no + 1
    
    # Calculate the average F1 score for the current configuration
    average_f1_score = np.mean(f1_scores)

    # Print the scores for the current configuration
    print(f"Configuration: filters={filter, filter*2}")
    print('Accuracy: %.3f (+/- %.3f)' % (np.mean(accuracy_scores), np.std(accuracy_scores)))
    print('Precision: %.3f (+/- %.3f)' % (np.mean(precision_scores), np.std(precision_scores)))
    print('Recall: %.3f (+/- %.3f)' % (np.mean(recall_scores), np.std(recall_scores)))
    print('F1 score: %.3f (+/- %.3f)' % (np.mean(f1_scores), np.std(f1_scores)))
    print()

    # Check if the current configuration has a higher average F1 score
    # If so, update the best configuration and best F1 score
    if average_f1_score > best_f1_score:
        best_configuration = (filter, filter*2)
        best_f1_score = average_f1_score


# Print the best configuration
print("Best Configuration of filters:")
print(best_configuration)
print("Best F1 Score:", best_f1_score)

#### RNN Concat + ML Models
def grid_rnn_concat(class_model):

    #insert the respective architecture configurations to be tested
    architectures = [
  
        {'hidden_layers': 2, 'units_per_layer': 8},
        {'hidden_layers': 1, 'units_per_layer': 16},
        {'hidden_layers': 3, 'units_per_layer': 8},
        {'hidden_layers': 1, 'units_per_layer': 32},
        {'hidden_layers': 2, 'units_per_layer': 16},
        {'hidden_layers': 3, 'units_per_layer': 16},
        {'hidden_layers': 1, 'units_per_layer':64},
        {'hidden_layers': 2, 'units_per_layer': 32},
        {'hidden_layers': 1, 'units_per_layer': 256},
        {'hidden_layers': 1, 'units_per_layer': 128},
        {'hidden_layers': 2, 'units_per_layer':64}
    ]

    best_architecture = None
    best_f1_score = 0.0
    
    for architecture in architectures:
        fold_no = 1
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for train_index, test_index in kf.split(X_train, y_train):
            
            X_train_dl, X_test_dl = X_train.iloc[train_index, :], X_train.iloc[test_index, :]
            y_train_dl, y_test_dl = y_train.iloc[train_index], y_train.iloc[test_index]

            stacked_train, X_train_static, y_train_dl, stacked_test,
            X_test_static, y_test_dl = preprocess_data(X_train_dl, y_train_dl, X_test_dl, y_test_dl)
            
            # Input layers
            num_time_steps, num_features = stacked_train.shape[1], stacked_train.shape[2]

            temporal_input = Input(shape=(num_time_steps, num_features), name='TEMPORAL_INPUT')
            static_input = Input(shape=(X_train_static.shape[1]), name='STATIC_INPUT')

            # RNN layers
            rnn_layer = SimpleRNN(architecture['units_per_layer'],
                                  return_sequences=True, name=f'RNN_LAYER_1')(temporal_input)
            for i in range(1, architecture['hidden_layers']):
                rnn_layer = SimpleRNN(architecture['units_per_layer'], 
                                      return_sequences=True, name=f'RNN_LAYER_{i + 1}')(rnn_layer)
            rnn_layer = Flatten(name='FLATTEN')(rnn_layer)

            # Concatenate RNN layer with static input
            RNN_combined = Concatenate(axis=1, name='RNN_CONCAT')([rnn_layer, static_input])
            output = Dense(1, activation='sigmoid', name='RNN_OUTPUT_LAYER')(RNN_combined)

            model = Model(inputs=[temporal_input, static_input], outputs=[output])

            initial_learning_rate = 0.0001
            decay_rate = 0.1
            decay_steps = 20
            batch_size = 64
            epochs = 50

            def learning_rate_scheduler(epoch):
                return initial_learning_rate * decay_rate ** (epoch // decay_steps)

            model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=initial_learning_rate),
                          metrics=['accuracy'])
            
            print('------------------------------------------------------------------------')
            print(f'Training for fold {fold_no}')

            lr_scheduler = LearningRateScheduler(learning_rate_scheduler)
            history = model.fit([stacked_train, X_train_static], y_train_dl,
                                                    epochs=epochs, batch_size=batch_size,
                                                    validation_data=([stacked_test, X_test_static], y_test_dl),
                                                    shuffle=False, callbacks=[lr_scheduler])

            # Plot the loss on train vs validate tests
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
            
            preds = classifier_prediction(stacked_train,stacked_test,X_train_static, X_test_static,
                                          y_train_dl,y_test_dl,model = class_model, 
                                          feature_extractor_model = model, layer_name='RNN_CONCAT')[0]
            
            accuracy = accuracy_score(y_test_dl, preds)
            precision = precision_score(y_test_dl, preds)
            recall = recall_score(y_test_dl, preds)
            f1 = f1_score(y_test_dl, preds)

            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            
            fold_no = fold_no + 1
            
        # Calculate the average F1 score for the current architecture
        average_f1_score = np.mean(f1_scores)

        # Print the scores for the current architecture
        print(f"Architecture: {architecture}")
        print('Accuracy: %.3f (+/- %.3f)' % (np.mean(accuracy_scores), np.std(accuracy_scores)))
        print('Precision: %.3f (+/- %.3f)' % (np.mean(precision_scores), np.std(precision_scores)))
        print('Recall: %.3f (+/- %.3f)' % (np.mean(recall_scores), np.std(recall_scores)))
        print('F1 score: %.3f (+/- %.3f)' % (np.mean(f1_scores), np.std(f1_scores)))
        print()
        
        # Check if the current architecture has a higher average F1 score
        # If so, update the best architecture and best F1 score
        if average_f1_score > best_f1_score:
            best_architecture = architecture
            best_f1_score = average_f1_score


    # Print the best architecture
    print("Best Architecture:")
    print(best_architecture)
    print("Best F1 Score:", best_f1_score)
    
grid_rnn_concat(LogisticRegression())
grid_rnn_concat(RandomForestClassifier())
grid_rnn_concat(XGBClassifier())

#### LSTM Concat + ML Models
def grid_lstm_concat(class_model):

   #insert the respective architecture configurations to be tested
    architectures = [
        {'hidden_layers': 4, 'units_per_layer': 64},
        {'hidden_layers': 5, 'units_per_layer': 32},
        {'hidden_layers': 6, 'units_per_layer': 32}
    ]

    best_architecture = None
    best_f1_score = 0.0

    for architecture in architectures:
        fold_no = 1
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for train_index, test_index in kf.split(X_train, y_train):
            
            X_train_dl, X_test_dl = X_train.iloc[train_index, :], X_train.iloc[test_index, :]
            y_train_dl, y_test_dl = y_train.iloc[train_index], y_train.iloc[test_index]

            stacked_train, X_train_static, y_train_dl, stacked_test, X_test_static, 
            y_test_dl = preprocess_data(X_train_dl, y_train_dl, X_test_dl, y_test_dl)
            
            # Input layers
            num_time_steps, num_features = stacked_train.shape[1], stacked_train.shape[2]

            temporal_input = Input(shape=(num_time_steps, num_features), name='TEMPORAL_INPUT')
            static_input = Input(shape=(X_train_static.shape[1]), name='STATIC_INPUT')

            # LSTM layers
            lstm_layer = LSTM(architecture['units_per_layer'], 
                              return_sequences=True, name=f'LSTM_LAYER_1')(temporal_input)
            for i in range(1, architecture['hidden_layers']):
                lstm_layer = LSTM(architecture['units_per_layer'], 
                                  return_sequences=True, name=f'LSTM_LAYER_{i + 1}')(lstm_layer)
            lstm_layer = Flatten(name='FLATTEN')(lstm_layer)

            # Concatenate LSTM layer with static input
            LSTM_combined = Concatenate(axis=1, name='LSTM_CONCAT')([lstm_layer, static_input])
            output = Dense(1, activation='sigmoid', name='LSTM_OUTPUT_LAYER')(LSTM_combined)

            model = Model(inputs=[temporal_input, static_input], outputs=[output])

            initial_learning_rate = 0.0001
            decay_rate = 0.1
            decay_steps = 20
            batch_size = 64
            epochs = 50

            def learning_rate_scheduler(epoch):
                return initial_learning_rate * decay_rate ** (epoch // decay_steps)

            model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=initial_learning_rate), 
                          metrics=['accuracy'])
            
            print('------------------------------------------------------------------------')
            print(f'Training for fold {fold_no}')

            lr_scheduler = LearningRateScheduler(learning_rate_scheduler)
            history = model.fit([stacked_train, X_train_static], y_train_dl,
                                                    epochs=epochs, batch_size=batch_size,
                                                    validation_data=([stacked_test, X_test_static], y_test_dl),
                                                    shuffle=False, callbacks=[lr_scheduler])

            # Plot the loss on train vs validate tests
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
            
            preds = classifier_prediction(stacked_train,stacked_test,X_train_static, X_test_static,
                                          y_train_dl,y_test_dl,model = class_model, 
                                          feature_extractor_model = model, layer_name='LSTM_CONCAT')[0]
            
            accuracy = accuracy_score(y_test_dl, preds)
            precision = precision_score(y_test_dl, preds)
            recall = recall_score(y_test_dl, preds)
            f1 = f1_score(y_test_dl, preds)

            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            
            fold_no = fold_no + 1
            
        # Calculate the average F1 score for the current architecture
        average_f1_score = np.mean(f1_scores)

        # Print the scores for the current architecture
        print(f"Architecture: {architecture}")
        print('Accuracy: %.3f (+/- %.3f)' % (np.mean(accuracy_scores), np.std(accuracy_scores)))
        print('Precision: %.3f (+/- %.3f)' % (np.mean(precision_scores), np.std(precision_scores)))
        print('Recall: %.3f (+/- %.3f)' % (np.mean(recall_scores), np.std(recall_scores)))
        print('F1 score: %.3f (+/- %.3f)' % (np.mean(f1_scores), np.std(f1_scores)))
        print()
        
        # Check if the current architecture has a higher average F1 score
        # If so, update the best architecture and best F1 score
        if average_f1_score > best_f1_score:
            best_architecture = architecture
            best_f1_score = average_f1_score


    # Print the best architecture
    print("Best Architecture:")
    print(best_architecture)
    print("Best F1 Score:", best_f1_score)
    
grid_lstm_concat(LogisticRegression())
grid_lstm_concat(RandomForestClassifier())
grid_lstm_concat(XGBClassifier())

#### CNN Concat + ML Models
def grid_cnn_concat(class_model):

    first_layer_filters = [32,64,128,256]

    best_configuration = None
    best_f1_score = 0.0

    for filter in first_layer_filters:
        
                
        fold_no = 1
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for train_index, test_index in kf.split(X_train, y_train):
            
            X_train_dl, X_test_dl = X_train.iloc[train_index,:], X_train.iloc[test_index,:]
            y_train_dl, y_test_dl = y_train.iloc[train_index], y_train.iloc[test_index]

            stacked_train, X_train_static, y_train_dl, stacked_test,
            X_test_static, y_test_dl = preprocess_data(X_train_dl,y_train_dl, X_test_dl, y_test_dl)
            
            num_time_steps, num_features = stacked_train.shape[1], stacked_train.shape[2]
            
            temporal_input = Input(shape=(num_time_steps, num_features), name='TEMPORAL_INPUT')
            static_input = Input(shape=(X_train_static.shape[1]), name='STATIC_INPUT')
            
            # CNN layers
            cnn_layer = Conv1D(filters=filter, kernel_size=5,
                               activation='relu',padding='same', name=f'CNN_LAYER_1')(temporal_input)
            cnn_layer = MaxPooling1D(pool_size=3)(cnn_layer)
            cnn_layer = Flatten(name='FLATTEN')(cnn_layer)

            # Concatenate CNN layer with static input
            cnn_combined = Concatenate(axis=1, name='CNN_CONCAT')([cnn_layer, static_input])
            output = Dense(1, activation='sigmoid', name='CNN_OUTPUT_LAYER')(cnn_combined)
            
            model = Model(inputs=[temporal_input, static_input], outputs=[output])
            
            initial_learning_rate = 0.001  # Initial learning rate
            decay_rate = 0.1  # Decay rate
            decay_steps = 20  # Decay steps (number of steps before applying decay)
            batch_size = 64
            epochs = 50

            def learning_rate_scheduler(epoch):
                return initial_learning_rate * decay_rate ** (epoch // decay_steps)
            
            model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate = initial_learning_rate), 
                          metrics=['accuracy'])
            
            
            print('------------------------------------------------------------------------')
            print(f'Training for fold {fold_no}')
            
            lr_scheduler = LearningRateScheduler(learning_rate_scheduler)
        
            # Train the model
            history = model.fit([stacked_train, X_train_static], y_train_dl,
                                                    epochs=epochs, batch_size=batch_size,
                                                    validation_data=([stacked_test, X_test_static], y_test_dl),
                                                    shuffle=False, callbacks=[lr_scheduler])
            
            # Plot the loss on train vs validate tests
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
            
            preds = classifier_prediction(stacked_train,stacked_test,X_train_static, X_test_static,
                                          y_train_dl,y_test_dl,model = class_model, 
                                          feature_extractor_model = model, layer_name='CNN_CONCAT')[0]
                
            accuracy = accuracy_score(y_test_dl, preds)
            precision = precision_score(y_test_dl, preds)
            recall = recall_score(y_test_dl, preds)
            f1 = f1_score(y_test_dl, preds)

            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            
            fold_no = fold_no + 1
        
        # Calculate the average F1 score for the current configuration
        average_f1_score = np.mean(f1_scores)
        total_params = model.count_params()

        # Print the scores for the current configuration
        print(f"Configuration: filters={filter}")
        print('Accuracy: %.3f (+/- %.3f)' % (np.mean(accuracy_scores), np.std(accuracy_scores)))
        print('Precision: %.3f (+/- %.3f)' % (np.mean(precision_scores), np.std(precision_scores)))
        print('Recall: %.3f (+/- %.3f)' % (np.mean(recall_scores), np.std(recall_scores)))
        print('F1 score: %.3f (+/- %.3f)' % (np.mean(f1_scores), np.std(f1_scores)))
        print("Total Trainable Parameters of CNN:", total_params)
        print()

        # Check if the current configuration has a higher average F1 score
        # If so, update the best configuration and best F1 score
        if average_f1_score > best_f1_score:
            best_configuration = (filter)
            best_f1_score = average_f1_score


    # Print the best configuration
    print("Best Configuration of filters:")
    print(best_configuration)
    print("Best F1 Score:", best_f1_score)
    
grid_cnn_concat(LogisticRegression())
grid_cnn_concat(RandomForestClassifier())
grid_cnn_concat(XGBClassifier())

## Elbow Curves for Optimal Architecture
#Excel file contains all the tables, so I retrieve the tables and then I plot the elbow curves 
# 1 file for RNN, LSTM and 1 file for CNN

paths = ["../elbow-plots/elbowplots_lstm_rnn.xlsx", "../elbow-plots/elbowplots_cnn.xlsx"]

for excel_file_path in paths:
    
    with pd.ExcelFile(excel_file_path) as xls:
        sheet_names = xls.sheet_names

    for sheet_name in sheet_names:
    
        data_excel = pd.read_excel(excel_file_path, sheet_name=sheet_name)
        data_excel = data_excel.rename(columns={'F1 Score': 'F1_score'})
        num_parameters = data_excel['Num_parameters']
        f1_scores = data_excel['F1_score']
            
        # Create the elbow plot
        plt.figure(figsize=(10, 6))
        plt.plot(num_parameters, f1_scores, marker='o')
        
        # Find the index of the best architecture
        best_index = data_excel['F1_score'].idxmax()
        
        # Highlight the point of the best architecture with a red dot
        plt.scatter(num_parameters[best_index], f1_scores[best_index], c='r', s=60)
        
            
        # Mark the point with the highest F1 score
        best_arch = data_excel.iloc[best_index]
        
        if "cnn" in excel_file_path:
            plt.text(0.97, 0.03, f'Best: Layers={int(best_arch["Layers"])}, Filters={best_arch["Filters"]}',
             transform=plt.gca().transAxes, color='black', fontsize=10,
             verticalalignment='bottom', horizontalalignment='right')
        else:
            plt.text(0.97, 0.03, f'Best: Layers={int(best_arch["Layers"])}, Units={int(best_arch["Units"])}',
                transform=plt.gca().transAxes, color='black', fontsize=10,
                verticalalignment='bottom', horizontalalignment='right')
            
        plt.xlabel('Number of Parameters')
        plt.ylabel('F1 Score')
        plt.title(f'Elbow Plot for {sheet_name}: F1 Score vs Number of Parameters')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
## Optimizer, Learning Rate, Batch Size Tuning
### Temporal Features

#### RNN Temporal (Dense)
#insert the respective configurations to be tested
initial_learning_rates = [0.00001]
optimizers = ['rmsprop']
batches = [32]

best_configuration = None
best_f1_score = 0.0

for lr in initial_learning_rates:
    for optimizer in optimizers:
        for bs in batches:
            
            fold_no = 1
            accuracy_scores = []
            precision_scores = []
            recall_scores = []
            f1_scores = []
        
            for train_index, test_index in kf.split(X_train, y_train):
                
                X_train_dl, X_test_dl = X_train.iloc[train_index,:], X_train.iloc[test_index,:]
                y_train_dl, y_test_dl = y_train.iloc[train_index], y_train.iloc[test_index]

                stacked_train, X_train_static, y_train_dl, stacked_test,
                X_test_static, y_test_dl = preprocess_data(X_train_dl,y_train_dl, X_test_dl, y_test_dl)
                
                num_time_steps, num_features = stacked_train.shape[1], stacked_train.shape[2]
                
                # Build the RNN model
                model = Sequential()
                model.add(SimpleRNN(32, return_sequences=True, input_shape=(num_time_steps, num_features)))
                model.add(SimpleRNN(32, return_sequences=True))
                model.add(SimpleRNN(32, return_sequences=True))
                model.add(Flatten())
                model.add(Dense(1, activation='sigmoid'))
                
                initial_learning_rate = lr  # Initial learning rate
                decay_rate = 0.1  # Decay rate
                decay_steps = 20  # Decay steps (number of steps before applying decay)
                epochs = 50

                def learning_rate_scheduler(epoch):
                    return initial_learning_rate * decay_rate ** (epoch // decay_steps)
                
                if optimizer == 'rmsprop':
                    optimizer = RMSprop(learning_rate=initial_learning_rate)
                elif optimizer == 'adam':
                    optimizer = Adam(learning_rate=initial_learning_rate)
                
                
                model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
                
                lr_scheduler = LearningRateScheduler(learning_rate_scheduler)
                
                print('------------------------------------------------------------------------')
                print(f'Training for fold {fold_no}')
            
                batch_size = bs
                
                # Train the model
                history = model.fit(stacked_train, y_train_dl, epochs=epochs, batch_size=batch_size,
                                    validation_data=(stacked_test, y_test_dl),shuffle=False,callbacks=[lr_scheduler])
                
                # Plot the loss on train vs validate tests
                plt.plot(history.history['loss'], label='Train Loss')
                plt.plot(history.history['val_loss'], label='Validation Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.show()
                
                y_pred_probs = model.predict(stacked_test)
                y_pred = (y_pred_probs>=0.5).astype(int)    
                
                accuracy =  accuracy_score(y_test_dl,y_pred)
                precision = precision_score(y_test_dl,y_pred)
                recall = recall_score(y_test_dl,y_pred)
                f1 =  f1_score(y_test_dl,y_pred)

                accuracy_scores.append(accuracy)
                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)
                
                fold_no = fold_no + 1
            
            # Calculate the average F1 score for the current configuration
            average_f1_score = np.mean(f1_scores)

            # Print the scores for the current configuration
            print(f"Configuration: Learning Rate={lr}, Optimizer={optimizer.get_config()['name']},Batch Size ={bs}")
            print('Accuracy: %.3f (+/- %.3f)' % (np.mean(accuracy_scores), np.std(accuracy_scores)))
            print('Precision: %.3f (+/- %.3f)' % (np.mean(precision_scores), np.std(precision_scores)))
            print('Recall: %.3f (+/- %.3f)' % (np.mean(recall_scores), np.std(recall_scores)))
            print('F1 score: %.3f (+/- %.3f)' % (np.mean(f1_scores), np.std(f1_scores)))
            print()
        
            # Check if the current configuration has a higher average F1 score
            # If so, update the best configuration and best F1 score
            if average_f1_score > best_f1_score:
                best_configuration = (lr, optimizer,batch_size)
                best_f1_score = average_f1_score


# Print the best configuration
print("Best Configuration (Learning Rate, Optimizer, Batch Size):")
print(best_configuration)
print("Best F1 Score:", best_f1_score)

#### LSTM Temporal (Dense)
#insert the respective configurations to be tested
initial_learning_rates = [0.001,0.00001]
optimizers = ['adam']
batches = [64]

best_configuration = None
best_f1_score = 0.0

for lr in initial_learning_rates:
    for optimizer in optimizers:
        for bs in batches:
            
            fold_no = 1
            accuracy_scores = []
            precision_scores = []
            recall_scores = []
            f1_scores = []
        
            for train_index, test_index in kf.split(X_train, y_train):
                
                X_train_dl, X_test_dl = X_train.iloc[train_index,:], X_train.iloc[test_index,:]
                y_train_dl, y_test_dl = y_train.iloc[train_index], y_train.iloc[test_index]

                stacked_train, X_train_static, y_train_dl, stacked_test, X_test_static,
                y_test_dl = preprocess_data(X_train_dl,y_train_dl, X_test_dl, y_test_dl)
                
                num_time_steps, num_features = stacked_train.shape[1], stacked_train.shape[2]
                
                # Build the LSTM model
                model = Sequential()
                model.add(LSTM(64, return_sequences=True, input_shape=(num_time_steps, num_features)))
                model.add(LSTM(64, return_sequences=True))
                model.add(LSTM(64, return_sequences=True))
                model.add(LSTM(64, return_sequences=True))
                model.add(Flatten())
                model.add(Dense(1, activation='sigmoid'))
                
                initial_learning_rate = lr  # Initial learning rate
                decay_rate = 0.1  # Decay rate
                decay_steps = 20  # Decay steps (number of steps before applying decay)
                epochs = 50

                def learning_rate_scheduler(epoch):
                    return initial_learning_rate * decay_rate ** (epoch // decay_steps)
                
                if optimizer == 'rmsprop':
                    optimizer = RMSprop(learning_rate=initial_learning_rate)
                elif optimizer == 'adam':
                    optimizer = Adam(learning_rate=initial_learning_rate)
                
                
                model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
                
                lr_scheduler = LearningRateScheduler(learning_rate_scheduler)
                
                print('------------------------------------------------------------------------')
                print(f'Training for fold {fold_no}')
            
                batch_size = bs
                
                # Train the model
                history = model.fit(stacked_train, y_train_dl, epochs=epochs, batch_size=batch_size,
                                    validation_data=(stacked_test, y_test_dl),shuffle=False,callbacks=[lr_scheduler])
                
                # Plot the loss on train vs validate tests
                plt.plot(history.history['loss'], label='Train Loss')
                plt.plot(history.history['val_loss'], label='Validation Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.show()
                
                y_pred_probs = model.predict(stacked_test)
                y_pred = (y_pred_probs>=0.5).astype(int)    
                
                accuracy =  accuracy_score(y_test_dl,y_pred)
                precision = precision_score(y_test_dl,y_pred)
                recall = recall_score(y_test_dl,y_pred)
                f1 =  f1_score(y_test_dl,y_pred)

                accuracy_scores.append(accuracy)
                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)
                
                fold_no = fold_no + 1
            
            # Calculate the average F1 score for the current configuration
            average_f1_score = np.mean(f1_scores)

            # Print the scores for the current configuration
            print(f"Configuration: Learning Rate={lr}, Optimizer={optimizer.get_config()['name']},Batch Size ={bs}")
            print('Accuracy: %.3f (+/- %.3f)' % (np.mean(accuracy_scores), np.std(accuracy_scores)))
            print('Precision: %.3f (+/- %.3f)' % (np.mean(precision_scores), np.std(precision_scores)))
            print('Recall: %.3f (+/- %.3f)' % (np.mean(recall_scores), np.std(recall_scores)))
            print('F1 score: %.3f (+/- %.3f)' % (np.mean(f1_scores), np.std(f1_scores)))
            print()
        
            # Check if the current configuration has a higher average F1 score
            # If so, update the best configuration and best F1 score
            if average_f1_score > best_f1_score:
                best_configuration = (lr, optimizer,batch_size)
                best_f1_score = average_f1_score


# Print the best configuration
print("Best Configuration (Learning Rate, Optimizer, Batch Size):")
print(best_configuration)
print("Best F1 Score:", best_f1_score)

#### CNN Temporal (Dense)
#insert the respective configurations to be tested
initial_learning_rates = [0.001]
optimizers = ['adam']
batches = [64]

best_configuration = None
best_f1_score = 0.0

for lr in initial_learning_rates:
    for optimizer in optimizers:
        for bs in batches:
            
            fold_no = 1
            accuracy_scores = []
            precision_scores = []
            recall_scores = []
            f1_scores = []
        
            for train_index, test_index in kf.split(X_train, y_train):
                
                X_train_dl, X_test_dl = X_train.iloc[train_index,:], X_train.iloc[test_index,:]
                y_train_dl, y_test_dl = y_train.iloc[train_index], y_train.iloc[test_index]

                stacked_train, X_train_static, y_train_dl, stacked_test, X_test_static, 
                y_test_dl = preprocess_data(X_train_dl,y_train_dl, X_test_dl, y_test_dl)
                
                num_time_steps, num_features = stacked_train.shape[1], stacked_train.shape[2]
                
                # Build the CNN model
                model = Sequential()
                model.add(Conv1D(filters=4, kernel_size=5, 
                                 activation='relu',padding='same', input_shape=(num_time_steps, num_features)))
                model.add(MaxPooling1D(pool_size=3))
                model.add(Conv1D(filters=8, padding='same', kernel_size=3, activation='relu'))
                model.add(MaxPooling1D(pool_size=2))
                model.add(Flatten())
                model.add(Dense(1, activation='sigmoid'))
                
                initial_learning_rate = lr  # Initial learning rate
                decay_rate = 0.1  # Decay rate
                decay_steps = 20  # Decay steps (number of steps before applying decay)
                epochs = 50

                def learning_rate_scheduler(epoch):
                    return initial_learning_rate * decay_rate ** (epoch // decay_steps)
                
                if optimizer == 'rmsprop':
                    optimizer = RMSprop(learning_rate=initial_learning_rate)
                elif optimizer == 'adam':
                    optimizer = Adam(learning_rate=initial_learning_rate)
                
                
                model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
                
                lr_scheduler = LearningRateScheduler(learning_rate_scheduler)
                
                print('------------------------------------------------------------------------')
                print(f'Training for fold {fold_no}')
            
                batch_size = bs
                
                # Train the model
                history = model.fit(stacked_train, y_train_dl, epochs=epochs, batch_size=batch_size,
                                    validation_data=(stacked_test, y_test_dl),shuffle=False,callbacks=[lr_scheduler])
                
                # Plot the loss on train vs validate tests
                plt.plot(history.history['loss'], label='Train Loss')
                plt.plot(history.history['val_loss'], label='Validation Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.show()
                
                y_pred_probs = model.predict(stacked_test)
                y_pred = (y_pred_probs>=0.5).astype(int)    
                
                accuracy =  accuracy_score(y_test_dl,y_pred)
                precision = precision_score(y_test_dl,y_pred)
                recall = recall_score(y_test_dl,y_pred)
                f1 =  f1_score(y_test_dl,y_pred)

                accuracy_scores.append(accuracy)
                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)
                
                fold_no = fold_no + 1
            
            # Calculate the average F1 score for the current configuration
            average_f1_score = np.mean(f1_scores)

            # Print the scores for the current configuration
            print(f"Configuration: Learning Rate={lr}, Optimizer={optimizer.get_config()['name']},Batch Size ={bs}")
            print('Accuracy: %.3f (+/- %.3f)' % (np.mean(accuracy_scores), np.std(accuracy_scores)))
            print('Precision: %.3f (+/- %.3f)' % (np.mean(precision_scores), np.std(precision_scores)))
            print('Recall: %.3f (+/- %.3f)' % (np.mean(recall_scores), np.std(recall_scores)))
            print('F1 score: %.3f (+/- %.3f)' % (np.mean(f1_scores), np.std(f1_scores)))
            print()
        
            # Check if the current configuration has a higher average F1 score
            # If so, update the best configuration and best F1 score
            if average_f1_score > best_f1_score:
                best_configuration = (lr, optimizer,batch_size)
                best_f1_score = average_f1_score


# Print the best configuration
print("Best Configuration (Learning Rate, Optimizer, Batch Size):")
print(best_configuration)
print("Best F1 Score:", best_f1_score)

#### RNN Temporal + ML Models
def grid_rnn_temporal_optim(class_model):
    
    #insert the respective configurations to be tested
    initial_learning_rates = [0.001,0.0001,0.00001]
    optimizers = ['adam']
    batches = [32,128]

    best_configuration = None
    best_f1_score = 0.0

    for lr in initial_learning_rates:
        for optimizer in optimizers:
            for bs in batches:
                
                fold_no = 1
                accuracy_scores = []
                precision_scores = []
                recall_scores = []
                f1_scores = []
                
                for train_index, test_index in kf.split(X_train, y_train):
            
                    X_train_dl, X_test_dl = X_train.iloc[train_index,:], X_train.iloc[test_index,:]
                    y_train_dl, y_test_dl = y_train.iloc[train_index], y_train.iloc[test_index]

                    stacked_train, X_train_static, y_train_dl, stacked_test,
                    X_test_static, y_test_dl = preprocess_data(X_train_dl,y_train_dl, X_test_dl, y_test_dl)
                    
                    num_time_steps, num_features = stacked_train.shape[1], stacked_train.shape[2]
                    
                    # Build the RNN model
                    model = Sequential()
                    model.add(SimpleRNN(32, return_sequences=True, input_shape=(num_time_steps, num_features)))
                    model.add(Flatten(name='RNN_FLATTEN'))
                    model.add(Dense(1, activation='sigmoid'))
                    
                    initial_learning_rate = lr  # Initial learning rate
                    decay_rate = 0.1  # Decay rate
                    decay_steps = 20  # Decay steps (number of steps before applying decay)
                    epochs = 50

                    def learning_rate_scheduler(epoch):
                        return initial_learning_rate * decay_rate ** (epoch // decay_steps)
                    
                    if optimizer == 'rmsprop':
                        optimizer = RMSprop(learning_rate=initial_learning_rate)
                    elif optimizer == 'adam':
                        optimizer = Adam(learning_rate=initial_learning_rate)
                    
                    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
                    
                    lr_scheduler = LearningRateScheduler(learning_rate_scheduler)
                    
                    print('------------------------------------------------------------------------')
                    print(f'Training for fold {fold_no}')
                
                    batch_size = bs
                
                    # Train the model
                    history = model.fit(stacked_train, y_train_dl, epochs=epochs, batch_size=batch_size,
                                        validation_data=(stacked_test, y_test_dl),shuffle=False,callbacks=[lr_scheduler])
                    
                    # Plot the loss on train vs validate tests
                    plt.plot(history.history['loss'], label='Train Loss')
                    plt.plot(history.history['val_loss'], label='Validation Loss')
                    plt.xlabel('Epochs')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.show()
                    
                    preds = classifier_prediction_temporal(stacked_train,stacked_test, y_train_dl, 
                                                           model = class_model, feature_extractor_model=model, 
                                                           layer_name='RNN_FLATTEN')[0]
                    
                    accuracy =  accuracy_score(y_test_dl,preds)
                    precision = precision_score(y_test_dl,preds)
                    recall = recall_score(y_test_dl,preds)
                    f1 =  f1_score(y_test_dl,preds)

                    accuracy_scores.append(accuracy)
                    precision_scores.append(precision)
                    recall_scores.append(recall)
                    f1_scores.append(f1)
                    
                    fold_no = fold_no + 1
                
                    
                # Calculate the average F1 score for the current configuration
                average_f1_score = np.mean(f1_scores)

                # Print the scores for the current configuration
                print(f"Configuration: Learning Rate={lr}, Optimizer={optimizer.get_config()['name']},Batch Size ={bs}")
                print('Accuracy: %.3f (+/- %.3f)' % (np.mean(accuracy_scores), np.std(accuracy_scores)))
                print('Precision: %.3f (+/- %.3f)' % (np.mean(precision_scores), np.std(precision_scores)))
                print('Recall: %.3f (+/- %.3f)' % (np.mean(recall_scores), np.std(recall_scores)))
                print('F1 score: %.3f (+/- %.3f)' % (np.mean(f1_scores), np.std(f1_scores)))
            
                # Check if the current configuration has a higher average F1 score
                # If so, update the best configuration and best F1 score
                if average_f1_score > best_f1_score:
                    best_configuration = (lr, optimizer,batch_size)
                    best_f1_score = average_f1_score
            
            # Print the best configuration
            print("Best Configuration (Learning Rate, Optimizer, Batch Size):")
            print(best_configuration)
            print("Best F1 Score:", best_f1_score)
            
grid_rnn_temporal_optim(LogisticRegression())
grid_rnn_temporal_optim(RandomForestClassifier())
grid_rnn_temporal_optim(XGBClassifier())

#### LSTM Temporal + ML Models
def grid_lstm_temporal_optim(class_model):
    
    #insert the respective configurations to be tested
    initial_learning_rates = [0.001, 0.00001]
    optimizers = ['adam']
    batches = [64]

    best_configuration = None
    best_f1_score = 0.0

    for lr in initial_learning_rates:
        for optimizer in optimizers:
            for bs in batches:
                
                fold_no = 1
                accuracy_scores = []
                precision_scores = []
                recall_scores = []
                f1_scores = []
        
                for train_index, test_index in kf.split(X_train, y_train):
                    
                    X_train_dl, X_test_dl = X_train.iloc[train_index,:], X_train.iloc[test_index,:]
                    y_train_dl, y_test_dl = y_train.iloc[train_index], y_train.iloc[test_index]

                    stacked_train, X_train_static, y_train_dl, stacked_test,
                    X_test_static, y_test_dl = preprocess_data(X_train_dl,y_train_dl, X_test_dl, y_test_dl)
                    
                    num_time_steps, num_features = stacked_train.shape[1], stacked_train.shape[2]
                    
                    # Build the LSTM model
                    model = Sequential()
                    model.add(LSTM(16, return_sequences=True, input_shape=(num_time_steps, num_features)))
                    model.add(Flatten(name='LSTM_FLATTEN'))
                    model.add(Dense(1, activation='sigmoid'))
                    
                    initial_learning_rate = lr  # Initial learning rate
                    decay_rate = 0.1  # Decay rate
                    decay_steps = 20  # Decay steps (number of steps before applying decay)
                    epochs = 50

                    def learning_rate_scheduler(epoch):
                        return initial_learning_rate * decay_rate ** (epoch // decay_steps)
                    
                    if optimizer == 'rmsprop':
                        optimizer = RMSprop(learning_rate=initial_learning_rate)
                    elif optimizer == 'adam':
                        optimizer = Adam(learning_rate=initial_learning_rate)
                    
                    
                    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=initial_learning_rate),
                                  metrics=['accuracy'])
                    
                    lr_scheduler = LearningRateScheduler(learning_rate_scheduler)
                
                    print('------------------------------------------------------------------------')
                    print(f'Training for fold {fold_no}')
                    
                    batch_size = bs                
                    
                    # Train the model
                    history = model.fit(stacked_train, y_train_dl, epochs=epochs, batch_size=batch_size,
                                        validation_data=(stacked_test, y_test_dl),shuffle=False,callbacks=[lr_scheduler])
                    
                    # Plot the loss on train vs validate tests
                    plt.plot(history.history['loss'], label='Train Loss')
                    plt.plot(history.history['val_loss'], label='Validation Loss')
                    plt.xlabel('Epochs')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.show()
                    
                    preds = classifier_prediction_temporal(stacked_train,stacked_test, y_train_dl,
                                                           model = class_model, feature_extractor_model=model,
                                                           layer_name='LSTM_FLATTEN')[0]
                    
                    accuracy =  accuracy_score(y_test_dl,preds)
                    precision = precision_score(y_test_dl,preds)
                    recall = recall_score(y_test_dl,preds)
                    f1 =  f1_score(y_test_dl,preds)

                    accuracy_scores.append(accuracy)
                    precision_scores.append(precision)
                    recall_scores.append(recall)
                    f1_scores.append(f1)
                    
                    fold_no = fold_no + 1
                    
                # Calculate the average F1 score for the current configuration
                average_f1_score = np.mean(f1_scores)

                # Print the scores for the current configuration
                print(f"Configuration: Learning Rate={lr}, Optimizer={optimizer.get_config()['name']},Batch Size ={bs}")
                print('Accuracy: %.3f (+/- %.3f)' % (np.mean(accuracy_scores), np.std(accuracy_scores)))
                print('Precision: %.3f (+/- %.3f)' % (np.mean(precision_scores), np.std(precision_scores)))
                print('Recall: %.3f (+/- %.3f)' % (np.mean(recall_scores), np.std(recall_scores)))
                print('F1 score: %.3f (+/- %.3f)' % (np.mean(f1_scores), np.std(f1_scores)))
                print()
                
                # Check if the current configuration has a higher average F1 score
                # If so, update the best configuration and best F1 score
                if average_f1_score > best_f1_score:
                    best_configuration = (lr, optimizer,batch_size)
                    best_f1_score = average_f1_score

    # Print the best configuration
    print("Best Architecture:")
    print(best_configuration)
    print("Best F1 Score:", best_f1_score)
    
grid_lstm_temporal_optim(LogisticRegression())
grid_lstm_temporal_optim(RandomForestClassifier())
grid_lstm_temporal_optim(XGBClassifier())

#### CNN Temporal + ML Models
def grid_cnn_temporal_optim(class_model):
    
    #insert the respective configurations to be tested
    initial_learning_rates = [0.001]
    optimizers = ['adam']
    batches = [64]

    best_configuration = None
    best_f1_score = 0.0

    for lr in initial_learning_rates:
        for optimizer in optimizers:
            for bs in batches:
                
                fold_no = 1
                accuracy_scores = []
                precision_scores = []
                recall_scores = []
                f1_scores = []
                
                for train_index, test_index in kf.split(X_train, y_train):
            
                    X_train_dl, X_test_dl = X_train.iloc[train_index,:], X_train.iloc[test_index,:]
                    y_train_dl, y_test_dl = y_train.iloc[train_index], y_train.iloc[test_index]

                    stacked_train, X_train_static, y_train_dl, stacked_test,
                    X_test_static, y_test_dl = preprocess_data(X_train_dl,y_train_dl, X_test_dl, y_test_dl)
                    
                    num_time_steps, num_features = stacked_train.shape[1], stacked_train.shape[2]
                    
                    # Build the CNN model
                    model = Sequential()
                    model.add(Conv1D(filters=16, kernel_size=5,
                                     activation='relu',padding='same', input_shape=(num_time_steps, num_features)))
                    model.add(MaxPooling1D(pool_size=3))
                    model.add(Conv1D(filters=32, padding='same', kernel_size=3, activation='relu'))
                    model.add(MaxPooling1D(pool_size=2))
                    model.add(Flatten(name = 'CNN_FLATTEN'))
                    model.add(Dense(1, activation='sigmoid'))
                    
                    initial_learning_rate = lr  # Initial learning rate
                    decay_rate = 0.1  # Decay rate
                    decay_steps = 20  # Decay steps (number of steps before applying decay)
                    epochs = 50

                    def learning_rate_scheduler(epoch):
                        return initial_learning_rate * decay_rate ** (epoch // decay_steps)
                    
                    if optimizer == 'rmsprop':
                        optimizer = RMSprop(learning_rate=initial_learning_rate)
                    elif optimizer == 'adam':
                        optimizer = Adam(learning_rate=initial_learning_rate)
                    
                  
                    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
                    
                    lr_scheduler = LearningRateScheduler(learning_rate_scheduler)
                    
                    print('------------------------------------------------------------------------')
                    print(f'Training for fold {fold_no}')
                
                    batch_size = bs
                
                    # Train the model
                    history = model.fit(stacked_train, y_train_dl, epochs=epochs, batch_size=batch_size,
                                        validation_data=(stacked_test, y_test_dl),shuffle=False,callbacks=[lr_scheduler])
                    
                    # Plot the loss on train vs validate tests
                    plt.plot(history.history['loss'], label='Train Loss')
                    plt.plot(history.history['val_loss'], label='Validation Loss')
                    plt.xlabel('Epochs')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.show()
                    
                    preds = classifier_prediction_temporal(stacked_train,stacked_test, y_train_dl, 
                                                           model = class_model, feature_extractor_model=model,
                                                           layer_name='CNN_FLATTEN')[0]
                    
                    accuracy =  accuracy_score(y_test_dl,preds)
                    precision = precision_score(y_test_dl,preds)
                    recall = recall_score(y_test_dl,preds)
                    f1 =  f1_score(y_test_dl,preds)

                    accuracy_scores.append(accuracy)
                    precision_scores.append(precision)
                    recall_scores.append(recall)
                    f1_scores.append(f1)
                    
                    fold_no = fold_no + 1
                    
                # Calculate the average F1 score for the current configuration
                average_f1_score = np.mean(f1_scores)

                # Print the scores for the current configuration
                print(f"Configuration: Learning Rate={lr}, Optimizer={optimizer.get_config()['name']},Batch Size ={bs}")
                print('Accuracy: %.3f (+/- %.3f)' % (np.mean(accuracy_scores), np.std(accuracy_scores)))
                print('Precision: %.3f (+/- %.3f)' % (np.mean(precision_scores), np.std(precision_scores)))
                print('Recall: %.3f (+/- %.3f)' % (np.mean(recall_scores), np.std(recall_scores)))
                print('F1 score: %.3f (+/- %.3f)' % (np.mean(f1_scores), np.std(f1_scores)))
            
                
                # Check if the current configuration has a higher average F1 score
                # If so, update the best configuration and best F1 score
                if average_f1_score > best_f1_score:
                    best_configuration = (lr, optimizer,batch_size)
                    best_f1_score = average_f1_score

            
            # Print the best configuration
            print("Best Configuration (Learning Rate, Optimizer, Batch Size):")
            print(best_configuration)
            print("Best F1 Score:", best_f1_score)
            
grid_cnn_temporal_optim(LogisticRegression())
grid_cnn_temporal_optim(RandomForestClassifier())
grid_cnn_temporal_optim(XGBClassifier())

### Static + Temporal features

#### RNN Concat (Dense)
#insert the respective configurations to be tested
initial_learning_rates = [0.001, 0.0001, 0.00001]
optimizers = ['rmsprop','adam']
batches = [128]

best_configuration = None
best_f1_score = 0.0

for lr in initial_learning_rates:
    for optimizer in optimizers:
        for bs in batches:

            fold_no = 1
            accuracy_scores = []
            precision_scores = []
            recall_scores = []
            f1_scores = []
    
            for train_index, test_index in kf.split(X_train, y_train):
                
                X_train_dl, X_test_dl = X_train.iloc[train_index, :], X_train.iloc[test_index, :]
                y_train_dl, y_test_dl = y_train.iloc[train_index], y_train.iloc[test_index]

                stacked_train, X_train_static, y_train_dl, stacked_test, X_test_static,
                y_test_dl = preprocess_data(X_train_dl, y_train_dl, X_test_dl, y_test_dl)
                
                # Input layers
                num_time_steps, num_features = stacked_train.shape[1], stacked_train.shape[2]

                temporal_input = Input(shape=(num_time_steps, num_features), name='TEMPORAL_INPUT')
                static_input = Input(shape=(X_train_static.shape[1]), name='STATIC_INPUT')

                # RNN layers
                rnn_layer = SimpleRNN(64, return_sequences=True, name=f'RNN_LAYER_1')(temporal_input)
                rnn_layer = SimpleRNN(64, return_sequences=True, name=f'RNN_LAYER_2')(rnn_layer)
                rnn_layer = SimpleRNN(64, return_sequences=True, name=f'RNN_LAYER_3')(rnn_layer)
                rnn_layer = Flatten(name='FLATTEN')(rnn_layer)

                # Concatenate RNN layer with static input
                rnn_combined = Concatenate(axis=1, name='rnn_CONCAT')([rnn_layer, static_input])
                output = Dense(1, activation='sigmoid', name='RNN_OUTPUT_LAYER')(rnn_combined)

                model = Model(inputs=[temporal_input, static_input], outputs=[output])

                initial_learning_rate = lr  # Initial learning rate
                decay_rate = 0.1  # Decay rate
                decay_steps = 20  # Decay steps (number of steps before applying decay)
                epochs = 50

                def learning_rate_scheduler(epoch):
                    return initial_learning_rate * decay_rate ** (epoch // decay_steps)
                
                if optimizer == 'rmsprop':
                    optimizer = RMSprop(learning_rate=initial_learning_rate)
                elif optimizer == 'adam':
                    optimizer = Adam(learning_rate=initial_learning_rate)

                model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
                
                lr_scheduler = LearningRateScheduler(learning_rate_scheduler)
                
                print('------------------------------------------------------------------------')
                print(f'Training for fold {fold_no}')
                
                batch_size = bs
                history = model.fit([stacked_train, X_train_static], y_train_dl,
                                                        epochs=epochs, batch_size=batch_size,
                                                        validation_data=([stacked_test, X_test_static], y_test_dl),
                                                        shuffle=False, callbacks=[lr_scheduler])

                # Plot the loss on train vs validate tests
                plt.plot(history.history['loss'], label='Train Loss')
                plt.plot(history.history['val_loss'], label='Validation Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.show()

                # Make predictions on the test set
                y_pred_probs = model.predict([stacked_test, X_test_static])
                y_pred = (y_pred_probs >= 0.5).astype(int)

                accuracy = accuracy_score(y_test_dl, y_pred)
                precision = precision_score(y_test_dl, y_pred)
                recall = recall_score(y_test_dl, y_pred)
                f1 = f1_score(y_test_dl, y_pred)

                accuracy_scores.append(accuracy)
                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)

                fold_no = fold_no + 1
            
            
            # Calculate the average F1 score for the current configuration
            average_f1_score = np.mean(f1_scores)

            # Print the scores for the current configuration
            print(f"Configuration: Learning Rate={lr}, Optimizer={optimizer.get_config()['name']},Batch Size ={bs}")
            print('Accuracy: %.3f (+/- %.3f)' % (np.mean(accuracy_scores), np.std(accuracy_scores)))
            print('Precision: %.3f (+/- %.3f)' % (np.mean(precision_scores), np.std(precision_scores)))
            print('Recall: %.3f (+/- %.3f)' % (np.mean(recall_scores), np.std(recall_scores)))
            print('F1 score: %.3f (+/- %.3f)' % (np.mean(f1_scores), np.std(f1_scores)))

            # Check if the current configuration has a higher average F1 score
            # If so, update the best configuration and best F1 score
            if average_f1_score > best_f1_score:
                best_configuration = (lr, optimizer,batch_size)
                best_f1_score = average_f1_score


# Print the best configuration
print("Best Configuration (Learning Rate, Optimizer, Batch Size):")
print(best_configuration)
print("Best F1 Score:", best_f1_score)
#### LSTM Concat (Dense)
#insert the respective configurations to be tested

initial_learning_rates = [0.001, 0.0001, 0.00001]
optimizers = ['rmsprop']
batches = [64]

best_configuration = None
best_f1_score = 0.0

for lr in initial_learning_rates:
    for optimizer in optimizers:
        for bs in batches:

            fold_no = 1
            accuracy_scores = []
            precision_scores = []
            recall_scores = []
            f1_scores = []
    
            for train_index, test_index in kf.split(X_train, y_train):
                
                X_train_dl, X_test_dl = X_train.iloc[train_index, :], X_train.iloc[test_index, :]
                y_train_dl, y_test_dl = y_train.iloc[train_index], y_train.iloc[test_index]

                stacked_train, X_train_static, y_train_dl, stacked_test, 
                X_test_static, y_test_dl = preprocess_data(X_train_dl, y_train_dl, X_test_dl, y_test_dl)
                
                # Input layers
                num_time_steps, num_features = stacked_train.shape[1], stacked_train.shape[2]

                temporal_input = Input(shape=(num_time_steps, num_features), name='TEMPORAL_INPUT')
                static_input = Input(shape=(X_train_static.shape[1]), name='STATIC_INPUT')

                # LSTM layers
                lstm_layer = LSTM(64, return_sequences=True, name=f'LSTM_LAYER_1')(temporal_input)
                lstm_layer = LSTM(64, return_sequences=True, name=f'LSTM_LAYER_2')(lstm_layer)
                lstm_layer = LSTM(64, return_sequences=True, name=f'LSTM_LAYER_3')(lstm_layer)
                lstm_layer = LSTM(64, return_sequences=True, name=f'LSTM_LAYER_4')(lstm_layer)
                lstm_layer = Flatten(name='FLATTEN')(lstm_layer)

                # Concatenate LSTM layer with static input
                lstm_combined = Concatenate(axis=1, name='LSTM_CONCAT')([lstm_layer, static_input])
                output = Dense(1, activation='sigmoid', name='LSTM_OUTPUT_LAYER')(lstm_combined)

                model = Model(inputs=[temporal_input, static_input], outputs=[output])

                initial_learning_rate = lr  # Initial learning rate
                decay_rate = 0.1  # Decay rate
                decay_steps = 20  # Decay steps (number of steps before applying decay)
                epochs = 50

                def learning_rate_scheduler(epoch):
                    return initial_learning_rate * decay_rate ** (epoch // decay_steps)
                
                if optimizer == 'rmsprop':
                    optimizer = RMSprop(learning_rate=initial_learning_rate)
                elif optimizer == 'adam':
                    optimizer = Adam(learning_rate=initial_learning_rate)

                model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
                
                lr_scheduler = LearningRateScheduler(learning_rate_scheduler)
                
                print('------------------------------------------------------------------------')
                print(f'Training for fold {fold_no}')

                batch_size = bs

                history = model.fit([stacked_train, X_train_static], y_train_dl,
                                                        epochs=epochs, batch_size=batch_size,
                                                        validation_data=([stacked_test, X_test_static], y_test_dl),
                                                        shuffle=False, callbacks=[lr_scheduler])

                # Plot the loss on train vs validate tests
                plt.plot(history.history['loss'], label='Train Loss')
                plt.plot(history.history['val_loss'], label='Validation Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.show()

                # Make predictions on the test set
                y_pred_probs = model.predict([stacked_test, X_test_static])
                y_pred = (y_pred_probs >= 0.5).astype(int)

                accuracy = accuracy_score(y_test_dl, y_pred)
                precision = precision_score(y_test_dl, y_pred)
                recall = recall_score(y_test_dl, y_pred)
                f1 = f1_score(y_test_dl, y_pred)

                accuracy_scores.append(accuracy)
                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)

                fold_no = fold_no + 1
            
            
            # Calculate the average F1 score for the current configuration
            average_f1_score = np.mean(f1_scores)

            # Print the scores for the current configuration
            print(f"Configuration: Learning Rate={lr}, Optimizer={optimizer.get_config()['name']},Batch Size ={bs}")
            print('Accuracy: %.3f (+/- %.3f)' % (np.mean(accuracy_scores), np.std(accuracy_scores)))
            print('Precision: %.3f (+/- %.3f)' % (np.mean(precision_scores), np.std(precision_scores)))
            print('Recall: %.3f (+/- %.3f)' % (np.mean(recall_scores), np.std(recall_scores)))
            print('F1 score: %.3f (+/- %.3f)' % (np.mean(f1_scores), np.std(f1_scores)))

            # Check if the current configuration has a higher average F1 score
            # If so, update the best configuration and best F1 score
            if average_f1_score > best_f1_score:
                best_configuration = (lr, optimizer,batch_size)
                best_f1_score = average_f1_score


# Print the best configuration
print("Best Configuration (Learning Rate, Optimizer, Batch Size):")
print(best_configuration)
print("Best F1 Score:", best_f1_score)

#### CNN Concat (Dense)
#insert the respective configurations to be tested
initial_learning_rates = [0.001, 0.0001, 0.00001]
optimizers = ['rmsprop','adam']
batches = [128]


best_configuration = None
best_f1_score = 0.0


for lr in initial_learning_rates:
    for optimizer in optimizers:
        for bs in batches:

            fold_no = 1
            accuracy_scores = []
            precision_scores = []
            recall_scores = []
            f1_scores = []
    
            for train_index, test_index in kf.split(X_train, y_train):
                
                X_train_dl, X_test_dl = X_train.iloc[train_index, :], X_train.iloc[test_index, :]
                y_train_dl, y_test_dl = y_train.iloc[train_index], y_train.iloc[test_index]

                stacked_train, X_train_static, y_train_dl, stacked_test,
                X_test_static, y_test_dl = preprocess_data(X_train_dl, y_train_dl, X_test_dl, y_test_dl)
                
                # Input layers
                num_time_steps, num_features = stacked_train.shape[1], stacked_train.shape[2]

                temporal_input = Input(shape=(num_time_steps, num_features), name='TEMPORAL_INPUT')
                static_input = Input(shape=(X_train_static.shape[1]), name='STATIC_INPUT')

                # CNN layers
                cnn_layer = Conv1D(filters=256, kernel_size=5, 
                                   activation='relu',padding='same', name=f'CNN_LAYER_1')(temporal_input)
                cnn_layer = MaxPooling1D(pool_size=3)(cnn_layer)
                cnn_layer = Flatten(name='FLATTEN')(cnn_layer)

                # Concatenate CNN layer with static input
                cnn_combined = Concatenate(axis=1, name='cnn_CONCAT')([cnn_layer, static_input])
                output = Dense(1, activation='sigmoid', name='cnn_OUTPUT_LAYER')(cnn_combined)

                model = Model(inputs=[temporal_input, static_input], outputs=[output])

                initial_learning_rate = lr  # Initial learning rate
                decay_rate = 0.1  # Decay rate
                decay_steps = 20  # Decay steps (number of steps before applying decay)
                epochs = 50

                def learning_rate_scheduler(epoch):
                    return initial_learning_rate * decay_rate ** (epoch // decay_steps)
                
                if optimizer == 'rmsprop':
                    optimizer = RMSprop(learning_rate=initial_learning_rate)
                elif optimizer == 'adam':
                    optimizer = Adam(learning_rate=initial_learning_rate)

                model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
                
                lr_scheduler = LearningRateScheduler(learning_rate_scheduler)
                
                print('------------------------------------------------------------------------')
                print(f'Training for fold {fold_no}')
                
                batch_size = bs
                
                # Train DL model
                history = model.fit([stacked_train, X_train_static], y_train_dl,
                                                        epochs=epochs, batch_size=batch_size,
                                                        validation_data=([stacked_test, X_test_static], y_test_dl),
                                                        shuffle=False, callbacks=[lr_scheduler])

                # Plot the loss on train vs validate tests
                plt.plot(history.history['loss'], label='Train Loss')
                plt.plot(history.history['val_loss'], label='Validation Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.show()

                # Make predictions on the test set
                y_pred_probs = model.predict([stacked_test, X_test_static])
                y_pred = (y_pred_probs >= 0.5).astype(int)

                accuracy = accuracy_score(y_test_dl, y_pred)
                precision = precision_score(y_test_dl, y_pred)
                recall = recall_score(y_test_dl, y_pred)
                f1 = f1_score(y_test_dl, y_pred)

                accuracy_scores.append(accuracy)
                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)

                fold_no = fold_no + 1
            
            
            # Calculate the average F1 score for the current configuration
            average_f1_score = np.mean(f1_scores)

            # Print the scores for the current configuration
            print(f"Configuration: Learning Rate={lr}, Optimizer={optimizer.get_config()['name']},Batch Size ={bs}")
            print('Accuracy: %.3f (+/- %.3f)' % (np.mean(accuracy_scores), np.std(accuracy_scores)))
            print('Precision: %.3f (+/- %.3f)' % (np.mean(precision_scores), np.std(precision_scores)))
            print('Recall: %.3f (+/- %.3f)' % (np.mean(recall_scores), np.std(recall_scores)))
            print('F1 score: %.3f (+/- %.3f)' % (np.mean(f1_scores), np.std(f1_scores)))

            # Check if the current configuration has a higher average F1 score
            # If so, update the best configuration and best F1 score
            if average_f1_score > best_f1_score:
                best_configuration = (lr, optimizer,batch_size)
                best_f1_score = average_f1_score

# Print the best configuration
print("Best Configuration (Learning Rate, Optimizer, Batch Size):")
print(best_configuration)
print("Best F1 Score:", best_f1_score)

#### RNN Concat + ML Models
def grid_rnn_concat_optim(class_model):
    
    #insert the respective configurations to be tested
    initial_learning_rates = [0.0001]
    optimizers = ['adam']
    batches = [128]

    best_configuration = None
    best_f1_score = 0.0

    for lr in initial_learning_rates:
        for optimizer in optimizers:
            for bs in batches:
                
                fold_no = 1
                accuracy_scores = []
                precision_scores = []
                recall_scores = []
                f1_scores = []
        
                for train_index, test_index in kf.split(X_train, y_train):
                    
                    X_train_dl, X_test_dl = X_train.iloc[train_index, :], X_train.iloc[test_index, :]
                    y_train_dl, y_test_dl = y_train.iloc[train_index], y_train.iloc[test_index]

                    stacked_train, X_train_static, y_train_dl, stacked_test, 
                    X_test_static, y_test_dl = preprocess_data(X_train_dl, y_train_dl, X_test_dl, y_test_dl)
                    
                    # Input layers
                    num_time_steps, num_features = stacked_train.shape[1], stacked_train.shape[2]

                    temporal_input = Input(shape=(num_time_steps, num_features), name='TEMPORAL_INPUT')
                    static_input = Input(shape=(X_train_static.shape[1]), name='STATIC_INPUT')

                    # RNN layers
                    rnn_layer = SimpleRNN(32, return_sequences=True, name=f'RNN_LAYER_1')(temporal_input)
                    rnn_layer = SimpleRNN(32, return_sequences=True, name=f'RNN_LAYER_2')(rnn_layer)
                    rnn_layer = Flatten(name='FLATTEN')(rnn_layer)

                    # Concatenate RNN layer with static input
                    RNN_combined = Concatenate(axis=1, name='RNN_CONCAT')([rnn_layer, static_input])
                    output = Dense(1, activation='sigmoid', name='RNN_OUTPUT_LAYER')(RNN_combined)

                    model = Model(inputs=[temporal_input, static_input], outputs=[output])

                    initial_learning_rate = lr  # Initial learning rate
                    decay_rate = 0.1  # Decay rate
                    decay_steps = 20  # Decay steps (number of steps before applying decay)
                    epochs = 50

                    def learning_rate_scheduler(epoch):
                        return initial_learning_rate * decay_rate ** (epoch // decay_steps)
                    
                    if optimizer == 'rmsprop':
                        optimizer = RMSprop(learning_rate=initial_learning_rate)
                    elif optimizer == 'adam':
                        optimizer = Adam(learning_rate=initial_learning_rate)

                    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
                    
                    lr_scheduler = LearningRateScheduler(learning_rate_scheduler)
                    
                    print('------------------------------------------------------------------------')
                    print(f'Training for fold {fold_no}')
                    
                    batch_size = bs

                    #Train DL model
                    history = model.fit([stacked_train, X_train_static], y_train_dl,
                                                            epochs=epochs, batch_size=batch_size,
                                                            validation_data=([stacked_test, X_test_static], y_test_dl),
                                                            shuffle=False, callbacks=[lr_scheduler])

                    # Plot the loss on train vs validate tests
                    plt.plot(history.history['loss'], label='Train Loss')
                    plt.plot(history.history['val_loss'], label='Validation Loss')
                    plt.xlabel('Epochs')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.show()
                    
                    preds = classifier_prediction(stacked_train,stacked_test,X_train_static, X_test_static,
                                                  y_train_dl,y_test_dl,model = class_model, 
                                                  feature_extractor_model = model, layer_name='RNN_CONCAT')[0]
                    
                    accuracy = accuracy_score(y_test_dl, preds)
                    precision = precision_score(y_test_dl, preds)
                    recall = recall_score(y_test_dl, preds)
                    f1 = f1_score(y_test_dl, preds)

                    accuracy_scores.append(accuracy)
                    precision_scores.append(precision)
                    recall_scores.append(recall)
                    f1_scores.append(f1)
                    
                    fold_no = fold_no + 1
                    
                # Calculate the average F1 score for the current configuration
                average_f1_score = np.mean(f1_scores)
             
                # Print the scores for the current configuration
                print(f"Configuration: Learning Rate={lr}, Optimizer={optimizer.get_config()['name']},Batch Size ={bs}")
                print('Accuracy: %.3f (+/- %.3f)' % (np.mean(accuracy_scores), np.std(accuracy_scores)))
                print('Precision: %.3f (+/- %.3f)' % (np.mean(precision_scores), np.std(precision_scores)))
                print('Recall: %.3f (+/- %.3f)' % (np.mean(recall_scores), np.std(recall_scores)))
                print('F1 score: %.3f (+/- %.3f)' % (np.mean(f1_scores), np.std(f1_scores)))
                
                # Check if the current configuration has a higher average F1 score
                # If so, update the best configuration and best F1 score
                if average_f1_score > best_f1_score:
                    best_configuration = (lr, optimizer,batch_size)
                    best_f1_score = average_f1_score


    # Print the best configuration
    print("Best Configuration (Learning Rate, Optimizer, Batch Size):")
    print(best_configuration)
    print("Best F1 Score:", best_f1_score)
    
grid_rnn_concat_optim(LogisticRegression())
grid_rnn_concat_optim(RandomForestClassifier())
grid_rnn_concat_optim(XGBClassifier())

#### LSTM Concat + ML Models
def grid_lstm_concat_optim(class_model):
    
    #insert the respective configurations to be tested
    initial_learning_rates = [0.001, 0.00001]
    optimizers = ['adam']
    batches = [64]

    best_configuration = None
    best_f1_score = 0.0

    for lr in initial_learning_rates:
        for optimizer in optimizers:
            for bs in batches:
                
                fold_no = 1
                accuracy_scores = []
                precision_scores = []
                recall_scores = []
                f1_scores = []
        
                for train_index, test_index in kf.split(X_train, y_train):
                    
                    X_train_dl, X_test_dl = X_train.iloc[train_index, :], X_train.iloc[test_index, :]
                    y_train_dl, y_test_dl = y_train.iloc[train_index], y_train.iloc[test_index]

                    stacked_train, X_train_static, y_train_dl, stacked_test,
                    X_test_static, y_test_dl = preprocess_data(X_train_dl, y_train_dl, X_test_dl, y_test_dl)
                    
                    # Input layers
                    num_time_steps, num_features = stacked_train.shape[1], stacked_train.shape[2]

                    temporal_input = Input(shape=(num_time_steps, num_features), name='TEMPORAL_INPUT')
                    static_input = Input(shape=(X_train_static.shape[1]), name='STATIC_INPUT')

                    # LSTM layers
                    lstm_layer = LSTM(128, return_sequences=True, name='LSTM_LAYER_1')(temporal_input)
                    lstm_layer = LSTM(128, return_sequences=True, name='LSTM_LAYER_2')(lstm_layer)
                    lstm_layer = Flatten(name='FLATTEN')(lstm_layer)

                    # Concatenate LSTM layer with static input
                    LSTM_combined = Concatenate(axis=1, name='LSTM_CONCAT')([lstm_layer, static_input])
                    output = Dense(1, activation='sigmoid', name='LSTM_OUTPUT_LAYER')(LSTM_combined)

                    model = Model(inputs=[temporal_input, static_input], outputs=[output])

                    initial_learning_rate = lr  # Initial learning rate
                    decay_rate = 0.1  # Decay rate
                    decay_steps = 20  # Decay steps (number of steps before applying decay)
                    epochs = 50

                    def learning_rate_scheduler(epoch):
                        return initial_learning_rate * decay_rate ** (epoch // decay_steps)
                    
                    if optimizer == 'rmsprop':
                        optimizer = RMSprop(learning_rate=initial_learning_rate)
                    elif optimizer == 'adam':
                        optimizer = Adam(learning_rate=initial_learning_rate)

                    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
                    
                    lr_scheduler = LearningRateScheduler(learning_rate_scheduler)
                    
                    print('------------------------------------------------------------------------')
                    print(f'Training for fold {fold_no}')
                    
                    batch_size = bs

                    #Train DL model
                    history = model.fit([stacked_train, X_train_static], y_train_dl,
                                                            epochs=epochs, batch_size=batch_size,
                                                            validation_data=([stacked_test, X_test_static], y_test_dl),
                                                            shuffle=False, callbacks=[lr_scheduler])

                    # Plot the loss on train vs validate tests
                    plt.plot(history.history['loss'], label='Train Loss')
                    plt.plot(history.history['val_loss'], label='Validation Loss')
                    plt.xlabel('Epochs')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.show()
                    
                    preds = classifier_prediction(stacked_train,stacked_test,X_train_static,
                                                  X_test_static,y_train_dl,y_test_dl,model = class_model, 
                                                  feature_extractor_model = model, layer_name='LSTM_CONCAT')[0]
                    
                    accuracy = accuracy_score(y_test_dl, preds)
                    precision = precision_score(y_test_dl, preds)
                    recall = recall_score(y_test_dl, preds)
                    f1 = f1_score(y_test_dl, preds)

                    accuracy_scores.append(accuracy)
                    precision_scores.append(precision)
                    recall_scores.append(recall)
                    f1_scores.append(f1)
                    
                    fold_no = fold_no + 1
                    
                # Calculate the average F1 score for the current architecture
                average_f1_score = np.mean(f1_scores)
             

                # Print the scores for the current configuration
                print(f"Configuration: Learning Rate={lr}, Optimizer={optimizer.get_config()['name']},Batch Size ={bs}")
                print('Accuracy: %.3f (+/- %.3f)' % (np.mean(accuracy_scores), np.std(accuracy_scores)))
                print('Precision: %.3f (+/- %.3f)' % (np.mean(precision_scores), np.std(precision_scores)))
                print('Recall: %.3f (+/- %.3f)' % (np.mean(recall_scores), np.std(recall_scores)))
                print('F1 score: %.3f (+/- %.3f)' % (np.mean(f1_scores), np.std(f1_scores)))
                
                # Check if the current configuration has a higher average F1 score
                # If so, update the best configuration and best F1 score
                if average_f1_score > best_f1_score:
                    best_configuration = (lr, optimizer,batch_size)
                    best_f1_score = average_f1_score


    # Print the best configuration
    print("Best Configuration (Learning Rate, Optimizer, Batch Size):")
    print(best_configuration)
    print("Best F1 Score:", best_f1_score)
    
grid_lstm_concat_optim(LogisticRegression())
grid_lstm_concat_optim(RandomForestClassifier())
grid_lstm_concat_optim(XGBClassifier())

#### CNN Concat + ML Models
def grid_cnn_concat_optim(class_model):
    
    #insert the respective configurations to be tested
    initial_learning_rates = [0.001]
    optimizers = ['adam']
    batches = [64]

    best_configuration = None
    best_f1_score = 0.0

    for lr in initial_learning_rates:
        for optimizer in optimizers:
            for bs in batches:
                
                fold_no = 1
                accuracy_scores = []
                precision_scores = []
                recall_scores = []
                f1_scores = []
        
                for train_index, test_index in kf.split(X_train, y_train):
                    
                    X_train_dl, X_test_dl = X_train.iloc[train_index, :], X_train.iloc[test_index, :]
                    y_train_dl, y_test_dl = y_train.iloc[train_index], y_train.iloc[test_index]

                    stacked_train, X_train_static, y_train_dl, stacked_test, 
                    X_test_static, y_test_dl = preprocess_data(X_train_dl, y_train_dl, X_test_dl, y_test_dl)
                    
                    # Input layers
                    num_time_steps, num_features = stacked_train.shape[1], stacked_train.shape[2]

                    temporal_input = Input(shape=(num_time_steps, num_features), name='TEMPORAL_INPUT')
                    static_input = Input(shape=(X_train_static.shape[1]), name='STATIC_INPUT')

                    # CNN layers
                    cnn_layer = Conv1D(filters=8, kernel_size=5,
                                       activation='relu',padding='same', name=f'CNN_LAYER_1')(temporal_input)
                    cnn_layer = MaxPooling1D(pool_size=3)(cnn_layer)
                    cnn_layer = Conv1D(filters=16, kernel_size=3,
                                       activation='relu',padding='same', name=f'CNN_LAYER_2')(cnn_layer)
                    cnn_layer = MaxPooling1D(pool_size=2)(cnn_layer)
                    cnn_layer = Flatten(name='FLATTEN')(cnn_layer)

                    # Concatenate CNN layer with static input
                    cnn_combined = Concatenate(axis=1, name='CNN_CONCAT')([cnn_layer, static_input])
                    output = Dense(1, activation='sigmoid', name='CNN_OUTPUT_LAYER')(cnn_combined)
                    
                    model = Model(inputs=[temporal_input, static_input], outputs=[output])

                    initial_learning_rate = lr  # Initial learning rate
                    decay_rate = 0.1  # Decay rate
                    decay_steps = 20  # Decay steps (number of steps before applying decay)
                    epochs = 50

                    def learning_rate_scheduler(epoch):
                        return initial_learning_rate * decay_rate ** (epoch // decay_steps)
                    
                    if optimizer == 'rmsprop':
                        optimizer = RMSprop(learning_rate=initial_learning_rate)
                    elif optimizer == 'adam':
                        optimizer = Adam(learning_rate=initial_learning_rate)

                    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
                    
                    lr_scheduler = LearningRateScheduler(learning_rate_scheduler)
                    
                    print('------------------------------------------------------------------------')
                    print(f'Training for fold {fold_no}')
                    
                    batch_size = bs

                    # Train DL model
                    history = model.fit([stacked_train, X_train_static], y_train_dl,
                                                            epochs=epochs, batch_size=batch_size,
                                                            validation_data=([stacked_test, X_test_static], y_test_dl),
                                                            shuffle=False, callbacks=[lr_scheduler])

                    # Plot the loss on train vs validate tests
                    plt.plot(history.history['loss'], label='Train Loss')
                    plt.plot(history.history['val_loss'], label='Validation Loss')
                    plt.xlabel('Epochs')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.show()
                    
                    preds = classifier_prediction(stacked_train,stacked_test,X_train_static,
                                                  X_test_static,y_train_dl,y_test_dl,model = class_model,
                                                  feature_extractor_model = model, layer_name='CNN_CONCAT')[0]
                    
                    accuracy = accuracy_score(y_test_dl, preds)
                    precision = precision_score(y_test_dl, preds)
                    recall = recall_score(y_test_dl, preds)
                    f1 = f1_score(y_test_dl, preds)

                    accuracy_scores.append(accuracy)
                    precision_scores.append(precision)
                    recall_scores.append(recall)
                    f1_scores.append(f1)
                    
                    fold_no = fold_no + 1
                    
                # Calculate the average F1 score for the current configuration
                average_f1_score = np.mean(f1_scores)
             

                # Print the scores for the current configuration
                print(f"Configuration: Learning Rate={lr}, Optimizer={optimizer.get_config()['name']},Batch Size ={bs}")
                print('Accuracy: %.3f (+/- %.3f)' % (np.mean(accuracy_scores), np.std(accuracy_scores)))
                print('Precision: %.3f (+/- %.3f)' % (np.mean(precision_scores), np.std(precision_scores)))
                print('Recall: %.3f (+/- %.3f)' % (np.mean(recall_scores), np.std(recall_scores)))
                print('F1 score: %.3f (+/- %.3f)' % (np.mean(f1_scores), np.std(f1_scores)))
                
                # Check if the current configuration has a higher average F1 score
                # If so, update the best configuration and best F1 score
                if average_f1_score > best_f1_score:
                    best_configuration = (lr, optimizer,batch_size)
                    best_f1_score = average_f1_score


    # Print the best configuration
    print("Best Configuration (Learning Rate, Optimizer, Batch Size):")
    print(best_configuration)
    print("Best F1 Score:", best_f1_score)
    
grid_cnn_concat_optim(LogisticRegression())
grid_cnn_concat_optim(RandomForestClassifier())
grid_cnn_concat_optim(XGBClassifier())
