from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
import pandas as pd 
from tkinter import simpledialog
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import keras

from keras import models

from keras import layers

from keras import utils

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures

from tensorflow.python.keras.models import model_from_json
import pickle
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from imblearn.over_sampling import SMOTE
#this is python code
main = tkinter.Tk()
main.title("Crop Yield Prediction")
main.geometry("1000x650")

global filename
global rnn_acc, lstm_acc, ff_acc
global classifier
global Y1
global rainfall_dataset
global crop_dataset
global le
scalerX = StandardScaler()

global weight_for_0
global weight_for_1

def upload():
    global filename
    global rainfall_dataset
    global crop_dataset
    global le
    filename = filedialog.askdirectory(initialdir = ".")
    rainfall_dataset = pd.read_csv('dataset/district wise rainfall normal.csv')
    crop_dataset = pd.read_csv('dataset/Agriculture In India.csv')
    crop_dataset.fillna(0, inplace = True)
    crop_dataset['Production'] = crop_dataset['Production'].astype(np.int64)
    print(crop_dataset.dtypes)
    print(crop_dataset['Production'])
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n\n')
    text.insert(END,str(crop_dataset.head))

        
def preprocess():
    global weight_for_0
    global weight_for_1
    global crop_dataset
    global le
    global X, Y
    text.delete('1.0', END)
    
    le = LabelEncoder()
    crop_dataset['State_Name'] = pd.Series(le.fit_transform(crop_dataset['State_Name']))
    crop_dataset['District_Name'] = pd.Series(le.fit_transform(crop_dataset['District_Name']))
    crop_dataset['Season'] = pd.Series(le.fit_transform(crop_dataset['Season']))
    crop_dataset['Crop'] = pd.Series(le.fit_transform(crop_dataset['Crop']))
    
    if 'Area' in crop_dataset.columns and 'Season' in crop_dataset.columns:
        crop_dataset['Area_Season'] = crop_dataset['Area'] * crop_dataset['Season']
    
    rainfall_exists = 'Rainfall' in crop_dataset.columns
    area_exists = 'Area' in crop_dataset.columns
    
    if rainfall_exists and area_exists:
        poly = PolynomialFeatures(2, include_bias=False)
        poly_features = poly.fit_transform(crop_dataset[['Area', 'Rainfall']])
        for i in range(poly_features.shape[1]):
            crop_dataset[f'Poly_Feature_{i}'] = poly_features[:, i]
    elif area_exists:
        # If only Area exists, create some simple derived features
        crop_dataset['Area_Squared'] = crop_dataset['Area'] ** 2
        crop_dataset['Area_Log'] = np.log1p(crop_dataset['Area'])  # log(1+x) to handle zeros
    
    # Continue with the rest of your preprocessing
    crop_datasets = crop_dataset.values
    cols = crop_datasets.shape[1]-1
    X = crop_datasets[:,0:cols]
    Y = crop_datasets[:,cols]
    Y = Y.astype('uint8')
    avg = np.average(Y)
    
    Y1 = []
    for i in range(len(Y)):
        if Y[i] >= avg:
            Y1.append(1)
        else:
            Y1.append(0)
    Y = np.asarray(Y1)
    Y = Y.astype('uint8')
    
    # Apply SMOTE to balance classes
    try:
        smote = SMOTE(random_state=42)
        X, Y = smote.fit_resample(X, Y)
        resampled = True
    except Exception as e:
        print(f"Error applying SMOTE: {e}")
        resampled = False
    
    a, b = np.unique(Y, return_counts=True)
    print(str(a)+" "+str(b))
    
    # Convert to categorical for neural network
    Y = utils.to_categorical(Y)
    Y = Y.astype('uint8')
    
    counts = np.bincount(np.argmax(Y, axis=1))
    weight_for_0 = 1.0 / counts[0] if counts[0] > 0 else 1.0
    weight_for_1 = 1.0 / counts[1] if counts[1] > 0 else 1.0
    
    print(X.shape)
    print(Y.shape)
    
    # Standardize features
    scalerX.fit(X)
    X = scalerX.transform(X)
    
    text.insert(END, "Data preprocessing completed.\n")
    text.insert(END, f"Dataset shape: {X.shape} features, {Y.shape[0]} samples\n")
    text.insert(END, "Class distribution:\n")
    if resampled:
        text.insert(END, "SMOTE resampling was applied to balance classes.\n")
    text.insert(END, f"Class 0: {counts[0]} samples\n")
    text.insert(END, f"Class 1: {counts[1]} samples\n")
    text.insert(END, "First few samples:\n")
    text.insert(END, str(X[:5]))

def runRNN():
    global rnn_acc
    global X, Y
    global classifier
    text.delete('1.0', END)
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=20,  
        min_delta=0.001,  
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.0001
    )
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    model = models.Sequential([
        layers.Dense(512, input_dim=X.shape[1], activation='relu', kernel_initializer="uniform"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu', kernel_initializer="uniform"),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu', kernel_initializer="uniform"),
        layers.Dense(Y.shape[1], activation='softmax', kernel_initializer="uniform")
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    print(model.summary())
    
    # Training without early stopping to ensure 5 epochs
    history = model.fit(
        X_train, Y_train,
        epochs=5,
        batch_size=64,
        validation_data=(X_val, Y_val),
        callbacks=[reduce_lr]  # Removed early stopping
    )
    
    # Get accuracy from best epoch
    values = history.history
    best_epoch = np.argmax(values['val_accuracy'])
    acc = values['val_accuracy'][best_epoch] * 100
    rnn_acc = acc
    
    # Save model and history
    f = open('model/rnnhistory.pckl', 'wb')
    pickle.dump(values, f)
    f.close()
    
    text.insert(END,'RNN Prediction Accuracy : '+str(acc)+"\n\n")
    text.insert(END,'Training completed in 5 epochs\n\n')
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(values['accuracy'])
    plt.plot(values['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(values['loss'])
    plt.plot(values['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.tight_layout()
    plt.savefig('rnn_training_history.png')
    
    classifier = model
    classifier.save_weights('model/rnnmodel_weights.h5')
    model_json = classifier.to_json()
    with open("model/rnnmodel.json", "w") as json_file:
        json_file.write(model_json)

def runLSTM():
    global lstm_acc
    global X, Y
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=20,  
        min_delta=0.001,  
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.0001
    )
    XX = X.reshape((X.shape[0], X.shape[1], 1))
    
    X_train, X_val, Y_train, Y_val = train_test_split(XX, Y, test_size=0.2, random_state=42)
    
    model = models.Sequential([
        keras.layers.LSTM(512, input_shape=(X.shape[1], 1), return_sequences=True),
        layers.Dropout(0.3),
        keras.layers.LSTM(256),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.Dense(Y.shape[1], activation='softmax')
    ])
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    print(model.summary())
    
    history = model.fit(
        X_train, Y_train,
        epochs=1,
        batch_size=64,
        validation_data=(X_val, Y_val),
        callbacks=[reduce_lr]  # Removed early stopping
    )
    
    # Get accuracy from best epoch
    values = history.history
    best_epoch = np.argmax(values['val_accuracy'])
    acc = values['val_accuracy'][best_epoch] * 100
    lstm_acc = acc
    
    # Save model and history
    f = open('model/lstmhistory.pckl', 'wb')
    pickle.dump(values, f)
    f.close()
    
    text.insert(END,'LSTM Prediction Accuracy : '+str(acc)+"\n\n")
    text.insert(END,'Training completed in 1 epochs\n\n')
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(values['accuracy'])
    plt.plot(values['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(values['loss'])
    plt.plot(values['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.tight_layout()
    plt.savefig('lstm_training_history.png')
    
    classifier1 = model
    classifier1.save_weights('model/lstmmodel_weights.h5')
    model_json = classifier1.to_json()
    with open("model/lstmmodel.json", "w") as json_file:
        json_file.write(model_json)

def runFF():
    global ff_acc
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=20,  
        min_delta=0.001, 
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.0001
    )
    
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(Y.shape[1], activation='softmax')])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    print(model.summary())
    
    history = model.fit(
        X_train, Y_train,
        epochs=5,
        batch_size=64,
        validation_data=(X_val, Y_val),
        callbacks=[reduce_lr]  
    )
    
    # Get accuracy from best epoch
    values = history.history
    best_epoch = np.argmax(values['val_accuracy'])
    ff_acc = values['val_accuracy'][best_epoch] * 100
    
    text.insert(END,'Feed Forward Neural Network Prediction Accuracy : '+str(ff_acc)+"\n\n")
    text.insert(END,'Training completed in 5 epochs\n\n')
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(values['accuracy'])
    plt.plot(values['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(values['loss'])
    plt.plot(values['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.tight_layout()
    plt.savefig('ff_training_history.png')

def graph():
    global rnn_acc, lstm_acc, ff_acc
    bars = ['RNN Accuracy','LSTM Accuracy','Feed Forward Accuracy']
    height = [rnn_acc, lstm_acc, ff_acc]
    y_pos = np.arange(len(bars))
    
    plt.figure(figsize=(10, 6))
    plt.bar(y_pos, height, color=['blue', 'green', 'orange'])
    plt.xticks(y_pos, bars)
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy Comparison')
    
    # Add value labels on top of each bar
    for i, v in enumerate(height):
        plt.text(i, v + 1, f"{v:.2f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png')
    plt.show()

def evaluate_model():
    # Function to perform k-fold cross validation
    global X, Y
    
    text.delete('1.0', END)
    text.insert(END, "Starting 5-fold cross-validation evaluation...\n\n")
    
    # Define models to evaluate
    models = {
        "RNN": create_rnn_model,
        "LSTM": create_lstm_model,
        "FF": create_ff_model
    }
    
    results = {}
    
    for model_name, model_builder in models.items():
        text.insert(END, f"Evaluating {model_name} model...\n")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_accuracies = []
        
        for fold, (train_index, val_index) in enumerate(kf.split(X)):
            X_train, X_val = X[train_index], X[val_index]
            Y_train, Y_val = Y[train_index], Y[val_index]
            
            if model_name == "LSTM":
                X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
            
            model = model_builder()
            
            # Define callbacks
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001
            )
            
            # Train model with fixed epochs
            history = model.fit(
                X_train, Y_train,
                epochs=5,
                batch_size=64,
                validation_data=(X_val, Y_val),
                callbacks=[reduce_lr],
                verbose=0
            )
            
            # Get best accuracy
            best_acc = max(history.history['val_accuracy']) * 100
            fold_accuracies.append(best_acc)
            text.insert(END, f"  Fold {fold+1} accuracy: {best_acc:.2f}%\n")
            
        # Calculate average accuracy across folds
        avg_acc = np.mean(fold_accuracies)
        std_acc = np.std(fold_accuracies)
        results[model_name] = avg_acc
        
        text.insert(END, f"{model_name} Cross-Validation Results:\n")
        text.insert(END, f"  Average accuracy: {avg_acc:.2f}% (±{std_acc:.2f}%)\n\n")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    bars = list(results.keys())
    height = list(results.values())
    y_pos = np.arange(len(bars))
    
    plt.bar(y_pos, height, color=['blue', 'green', 'orange'])
    plt.xticks(y_pos, bars)
    plt.ylabel('Average Accuracy (%)')
    plt.title('Cross-Validation Model Accuracy Comparison')
    
    # Add value labels on top of each bar
    for i, v in enumerate(height):
        plt.text(i, v + 1, f"{v:.2f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig('cross_validation_comparison.png')
    plt.show()

def create_rnn_model():
    # Function to create RNN model
    model = models.Sequential([
        layers.Dense(512, input_dim=X.shape[1], activation='relu', kernel_initializer="uniform"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu', kernel_initializer="uniform"),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu', kernel_initializer="uniform"),
        layers.Dense(Y.shape[1], activation='softmax', kernel_initializer="uniform")
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

def create_lstm_model():
    # Function to create LSTM model
    model = models.Sequential([
        keras.layers.LSTM(512, input_shape=(X.shape[1], 1), return_sequences=True),
        layers.Dropout(0.3),
        keras.layers.LSTM(256),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.Dense(Y.shape[1], activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

def create_ff_model():
    # Function to create Feed Forward model
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(Y.shape[1], activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

# UI setup
font = ('times', 15, 'bold')
title = Label(main, text='Crop Yield Prediction', justify=LEFT)
title.config(bg='lavender blush', fg='DarkOrchid1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Agriculture Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocess)
preprocessButton.place(x=300,y=100)
preprocessButton.config(font=font1)

rnnButton = Button(main, text="Run RNN Algorithm", command=runRNN)
rnnButton.place(x=500,y=100)
rnnButton.config(font=font1)

lstmButton = Button(main, text="Run LSTM Algorithm", command=runLSTM)
lstmButton.place(x=700,y=100)
lstmButton.config(font=font1)

ffButton = Button(main, text="Run Feedforward Neural Network", command=runFF)
ffButton.place(x=50,y=150)
ffButton.config(font=font1)

graphButton = Button(main, text="Accuracy Comparison Graph", command=graph)
graphButton.place(x=300,y=150)
graphButton.config(font=font1)

evalButton = Button(main, text="Cross-Validation Evaluation", command=evaluate_model)
evalButton.place(x=550,y=150)
evalButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=160)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=200)
text.config(font=font1) 

main.config(bg='light coral')
main.mainloop()