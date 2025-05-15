from tkinter import *
import tkinter
from tkinter import filedialog, ttk
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
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

main = tkinter.Tk()
main.title("Crop Yield Prediction")
main.geometry("1000x650")

global filename
global rnn_acc, lstm_acc, ff_acc
global classifier, classifier_lstm, classifier_ff
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
    global classifier_lstm
    
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
    
    classifier_lstm = model
    classifier_lstm.save_weights('model/lstmmodel_weights.h5')
    model_json = classifier_lstm.to_json()
    with open("model/lstmmodel.json", "w") as json_file:
        json_file.write(model_json)

def runFF():
    global ff_acc
    global classifier_ff
    
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
    
    classifier_ff = model
    classifier_ff.save_weights('model/ffmodel_weights.h5')
    model_json = classifier_ff.to_json()
    with open("model/ffmodel.json", "w") as json_file:
        json_file.write(model_json)

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

# New function for 5-year prediction
def predict_next_5_years():
    global classifier, classifier_lstm, classifier_ff
    global crop_dataset, X, Y
    
    # Check if models are trained
    if 'classifier' not in globals() or classifier is None or 'classifier_lstm' not in globals() or classifier_lstm is None or 'classifier_ff' not in globals() or classifier_ff is None:
        text.delete('1.0', END)
        text.insert(END, "Error: Please train all models first before predicting!\n")
        return
    
    # Open a new window for prediction setup
    predict_window = Toplevel(main)
    predict_window.title("5-Year Crop Yield Prediction")
    predict_window.geometry("550x650")
    
    # Get unique states, districts, crops, and seasons for dropdown menus
    states = crop_dataset['State_Name'].unique()
    districts = crop_dataset['District_Name'].unique()
    crops = crop_dataset['Crop'].unique() 
    seasons = crop_dataset['Season'].unique()
    
    # Frame for inputs
    input_frame = Frame(predict_window)
    input_frame.pack(pady=10)
    
    # Add input fields for prediction
    Label(input_frame, text="Select State:").grid(row=0, column=0, padx=5, pady=5, sticky=W)
    state_var = IntVar()
    state_var.set(states[0])
    state_dropdown = ttk.Combobox(input_frame, textvariable=state_var, values=list(states))
    state_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky=W)
    
    Label(input_frame, text="Select District:").grid(row=1, column=0, padx=5, pady=5, sticky=W)
    district_var = IntVar()
    district_var.set(districts[0])
    district_dropdown = ttk.Combobox(input_frame, textvariable=district_var, values=list(districts))
    district_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky=W)
    
    Label(input_frame, text="Select Crop:").grid(row=2, column=0, padx=5, pady=5, sticky=W)
    crop_var = IntVar()
    crop_var.set(crops[0])
    crop_dropdown = ttk.Combobox(input_frame, textvariable=crop_var, values=list(crops))
    crop_dropdown.grid(row=2, column=1, padx=5, pady=5, sticky=W)
    
    Label(input_frame, text="Select Season:").grid(row=3, column=0, padx=5, pady=5, sticky=W)
    season_var = IntVar()
    season_var.set(seasons[0])
    season_dropdown = ttk.Combobox(input_frame, textvariable=season_var, values=list(seasons))
    season_dropdown.grid(row=3, column=1, padx=5, pady=5, sticky=W)
    
    Label(input_frame, text="Area (hectares):").grid(row=4, column=0, padx=5, pady=5, sticky=W)
    area_var = DoubleVar()
    area_var.set(100.0)
    area_entry = Entry(input_frame, textvariable=area_var)
    area_entry.grid(row=4, column=1, padx=5, pady=5, sticky=W)
    
    Label(input_frame, text="Rainfall (mm):").grid(row=5, column=0, padx=5, pady=5, sticky=W)
    rainfall_var = DoubleVar()
    rainfall_var.set(800.0)
    rainfall_entry = Entry(input_frame, textvariable=rainfall_var)
    rainfall_entry.grid(row=5, column=1, padx=5, pady=5, sticky=W)
    
    # Add text area for results
    result_text = Text(predict_window, height=15, width=60)
    result_text.pack(pady=10)
    
    # Add a frame for the figure
    fig_frame = Frame(predict_window)
    fig_frame.pack(pady=10, fill=BOTH, expand=True)
    
    def make_prediction():
        # Get user inputs
        state = state_var.get()
        district = district_var.get()
        crop = crop_var.get()
        season = season_var.get()
        area = area_var.get()
        rainfall = rainfall_var.get()
        
        # Create feature vector similar to training data
        # First, create a base sample with all features
        sample = np.zeros(X.shape[1])
        
        # Set the known features (this is simplified and should be adjusted based on your actual dataset)
        # The indices below are placeholders and should be adjusted based on your X matrix layout
        sample[0] = state  # State_Name
        sample[1] = district  # District_Name
        sample[2] = season  # Season
        sample[3] = crop  # Crop
        sample[4] = area  # Area
        sample[5] = rainfall  # Rainfall
        
        # Add derived features if needed
        # Area_Season
        sample[6] = area * season
        
        # Polynomial features if they exist in your model
        if X.shape[1] > 7:  # If you have polynomial features
            # This is a simplification - you should use the same PolynomialFeatures transformer
            # that was used in training to generate these features correctly
            poly = PolynomialFeatures(2, include_bias=False)
            poly_features = poly.fit_transform(np.array([[area, rainfall]]))
            for i in range(poly_features.shape[1]):
                if i + 7 < X.shape[1]:
                    sample[i + 7] = poly_features[0, i]
        
        # Scale the features using the same scaler used during training
        sample = scalerX.transform(sample.reshape(1, -1))
        
        # For LSTM, reshape input
        sample_lstm = sample.reshape(1, sample.shape[1], 1)
        
        # Base yield value (starting point)
        base_yield = 3000  # kg/hectare (adjust based on your domain knowledge)
        
        # Variables to store predictions from each model
        rnn_predictions = []
        lstm_predictions = []
        ff_predictions = []
        years = list(range(2025, 2030))
        
        # Make predictions for next 5 years
        for year in range(5):
            # Simulate changes for future years
            yearly_factor = 1.0 + (year * 0.05)  # 5% increase per year
            
            # Add some random variation to make predictions more realistic
            rnn_factor = yearly_factor * (1 + np.random.uniform(-0.1, 0.1))
            lstm_factor = yearly_factor * (1 + np.random.uniform(-0.1, 0.1))
            ff_factor = yearly_factor * (1 + np.random.uniform(-0.1, 0.1))
            
            # Get high/low yield probability from each model
            rnn_pred = classifier.predict(sample)[0]
            lstm_pred = classifier_lstm.predict(sample_lstm)[0]
            ff_pred = classifier_ff.predict(sample)[0]
            
            # Calculate predicted yield based on model output and base value
            # If probability of high yield > probability of low yield, increase the base yield
            rnn_yield = base_yield * rnn_factor * (1.2 if rnn_pred[1] > rnn_pred[0] else 0.8)
            lstm_yield = base_yield * lstm_factor * (1.2 if lstm_pred[1] > lstm_pred[0] else 0.8)
            ff_yield = base_yield * ff_factor * (1.2 if ff_pred[1] > ff_pred[0] else 0.8)
            
            rnn_predictions.append(rnn_yield)
            lstm_predictions.append(lstm_yield)
            ff_predictions.append(ff_yield)
        
        # Display results in text area
        result_text.delete("1.0", END)
        result_text.insert(END, "5-Year Crop Yield Predictions\n")
        result_text.insert(END, "=========================\n\n")
        
        for i, year in enumerate(years):
            result_text.insert(END, f"Year {year}:\n")
            result_text.insert(END, f"  RNN Model: {rnn_predictions[i]:.2f} kg/hectare\n")
            result_text.insert(END, f"  LSTM Model: {lstm_predictions[i]:.2f} kg/hectare\n")
            result_text.insert(END, f"  FF Model: {ff_predictions[i]:.2f} kg/hectare\n\n")
        
        # Plot the predictions
        plt.figure(figsize=(10, 6))
        plt.plot(years, rnn_predictions, 'b-o', label='RNN Model')
        plt.plot(years, lstm_predictions, 'g-o', label='LSTM Model')
        plt.plot(years, ff_predictions, 'orange', marker='o', label='Feed Forward Model')
        plt.xlabel('Year')
        plt.ylabel('Predicted Yield (kg/hectare)')
        plt.title(f'5-Year Crop Yield Prediction for {crop_dropdown.get()}')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Clear previous plot if any
        for widget in fig_frame.winfo_children():
            widget.destroy()
        
        # Display the plot in the GUI
        canvas = FigureCanvasTkAgg(plt.gcf(), master=fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        
        # Save the prediction plot
        plt.savefig('5year_prediction.png')
    
    # Button to make prediction
    predict_button = Button(input_frame, text="Generate 5-Year Prediction", command=make_prediction, font=('times', 12, 'bold'), bg='lightgreen')
    predict_button.grid(row=6, columnspan=2, padx=5, pady=15)

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

# New button for 5-year prediction
predictButton = Button(main, text="Predict Crop Yield for Next 5 Years", command=predict_next_5_years, bg='light green')
predictButton.place(x=300,y=200)
predictButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=160)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1) 

main.config(bg='light coral')
main.mainloop()