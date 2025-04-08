#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.serialization import add_safe_globals
from sklearn.impute import SimpleImputer
import sklearn

from helper_code import *

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose):
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Extract the features and labels from the data.
    if verbose:
        print('Extracting features and labels from the data...')

    features = np.zeros((num_records, 6), dtype=np.float64)
    labels = np.zeros(num_records, dtype=bool)

    # Iterate over the records.
    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])
        features[i] = extract_features(record)
        labels[i] = load_label(record)

    # Train the models.
    # if verbose:
    #     print('Training the model on the data...')

    # # This very simple model trains a random forest model with very simple features.

    # # Define the parameters for the random forest classifier and regressor.
    # n_estimators = 12  # Number of trees in the forest.
    # max_leaf_nodes = 34  # Maximum number of leaf nodes in each tree.
    # random_state = 56  # Random state; set for reproducibility.

    # # Fit the model.
    # model = RandomForestClassifier(
    #     n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, labels)

    # # Create a folder for the model if it does not already exist.
    # os.makedirs(model_folder, exist_ok=True)

    # # Save the model.
    # save_model(model_folder, model)
    
    ### NEW CODE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imputer = SimpleImputer(strategy='mean')
    features = imputer.fit_transform(features)
    
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(1)  # shape: (N, 1, 6)
    labels_tensor = torch.tensor(labels.astype(np.float32)) # Convert boolean labels to floats (0.0 or 1.0)
    
    features_tensor = features_tensor.to(device)
    labels_tensor = labels_tensor.to(device)

    if verbose:
        print(f'Training the model on the data using {device}...')
    
    model = CNNModel().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    num_epochs = 50
        
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(features_tensor).squeeze(1)  # shape: (N,)
        loss = criterion(outputs, labels_tensor)
        loss.backward()
        optimizer.step()
        
        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
            
    model_to_save = {
        "state_dict": model.state_dict(),
        "imputer": imputer
    }
    save_model(model_folder, model_to_save)
                                 
    if verbose:
        print('Done.')
        print()

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    # model_filename = os.path.join(model_folder, 'model.sav')
    # model = joblib.load(model_filename)
    # return model

    ## NEW CODE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filepath = os.path.join(model_folder, "cnn_model.pth")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found in {model_folder}")
    
    with torch.serialization.safe_globals([sklearn.impute.SimpleImputer, np.dtype]):
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    
    #checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model_instance = CNNModel()
    model_instance.load_state_dict(checkpoint["state_dict"])
    model_instance.to(device)
    model_instance.eval()
    model_instance.imputer = checkpoint["imputer"]
    

    
    if verbose:
        print(f"Model loaded from {filepath} on device: {device}")
    
    return model_instance

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    # Load the model.
    # model = model['model']

    # # Extract the features.
    # features = extract_features(record)
    # features = features.reshape(1, -1)

    # # Get the model outputs.
    # binary_output = model.predict(features)[0]
    # probability_output = model.predict_proba(features)[0][1]
    
    ## NEW CODE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract the features.
    features = extract_features(record)
    features = features.reshape(1, -1)  # shape: (1,6)
    
    # Use the saved imputer to fill in missing values.
    features = model.imputer.transform(features)
    
    # Reshape features for the CNN: (batch_size, channels, sequence_length)
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(1)  # shape: (1, 1, 6)
    features_tensor = features_tensor.to(device)
    
    # Get the model outputs.
    with torch.no_grad():
        output = model(features_tensor)
    
    probability_output = output.item()
    binary_output = (probability_output >= 0.5)
    
    if verbose:
        print(f"Record: {record}\nPredicted Probability: {probability_output:.4f}, Binary Output: {binary_output}")
    
    return binary_output, probability_output

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract your features.
def extract_features(record):
    header = load_header(record)
    age = get_age(header)
    sex = get_sex(header)

    one_hot_encoding_sex = np.zeros(3, dtype=bool)
    if sex == 'Female':
        one_hot_encoding_sex[0] = 1
    elif sex == 'Male':
        one_hot_encoding_sex[1] = 1
    else:
        one_hot_encoding_sex[2] = 1

    signal, fields = load_signals(record)

    # TO-DO: Update to compute per-lead features. Check lead order and update and use functions for reordering leads as needed.

    num_finite_samples = np.size(np.isfinite(signal))
    if num_finite_samples > 0:
        signal_mean = np.nanmean(signal)
    else:
        signal_mean = 0.0
    if num_finite_samples > 1:
        signal_std = np.nanstd(signal)
    else:
        signal_std = 0.0

    features = np.concatenate(([age], one_hot_encoding_sex, [signal_mean, signal_std]))

    return np.asarray(features, dtype=np.float32)

# Save your trained model.
def save_model(model_folder, model):
    # d = {'model': model}
    # filename = os.path.join(model_folder, 'model.sav')
    # joblib.dump(d, filename, protocol=0)
    
    ##NEW CODE
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    filepath = os.path.join(model_folder, "cnn_model.pth")
    torch.save(model, filepath)
    
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Input: (batch, 1, 6)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3)  # output: (batch, 8, 4) because 6 - 3 + 1 = 4
        self.relu = nn.ReLU()
        self.fc = nn.Linear(8 * 4, 1)  # flatten the conv output and then use a linear layer
        self.sigmoid = nn.Sigmoid()
        self.imputer = None
    
    def forward(self, x):
        # x shape: (batch, 1, 6)
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # flatten to (batch, 8*4)
        x = self.fc(x)
        x = self.sigmoid(x)  # output probability between 0 and 1
        return x