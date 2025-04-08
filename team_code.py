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
import scipy.signal
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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
    # if verbose:
    #     print('Finding the Challenge data...')

    # records = find_records(data_folder)
    # num_records = len(records)

    # if num_records == 0:
    #     raise FileNotFoundError('No data were provided.')

    # # Extract the features and labels from the data.
    # if verbose:
    #     print('Extracting features and labels from the data...')

    # features = np.zeros((num_records, 4000, 12), dtype=np.float32)
    # labels = np.zeros(num_records, dtype=bool)

    # # Iterate over the records.
    # k=0
    # for i in range(num_records):
    #     if verbose:
    #         width = len(str(num_records))
    #         print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

    #     record = os.path.join(data_folder, records[i])
    #     features[i] = extract_features(record)
    #     labels[i] = load_label(record)

    #     k+=1
    #     if(k==100):
    #         break

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

    # Create dataset and dataloader.
    dataset = ChallengeDataset(data_folder, verbose=verbose)
    num_records = len(dataset)
    if num_records == 0:
        raise FileNotFoundError("No data were provided.")
    
    
    batch_size = 64
    num_epochs = 10
    learning_rate = 0.001
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    if verbose:
        print(f"Training on {num_records} records in batches of {batch_size}...")

    if verbose:
        print(f'Training the model on the data using {device}...')
    
    model = CNNModel().to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
        
    model.train()
    for epoch in tqdm(range(num_epochs), desc="Training Epochs", unit="epoch"):
        running_loss = 0.0
        batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch", leave=False)
        for batch_features, batch_labels in batch_pbar:
            batch_features = batch_features.to(device)  # shape: (batch, 12, 4000)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)  # shape: (batch, 2)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_features.size(0)
        
        epoch_loss = running_loss / num_records
        if verbose: # and ((epoch + 1) % 10 == 0 or epoch == 0):
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
            
    model_to_save = {"state_dict": model.state_dict()}
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
    
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    
    #checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model_instance = CNNModel().to(device)
    model_instance.load_state_dict(checkpoint["state_dict"])
    model_instance.eval()
    
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
    features = np.transpose(features, (1, 0)) # shape: (12,4000)

    # Reshape features for the CNN: (batch_size, channels, sequence_length)
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device) # shape: (1, 12, 4000)
    features_tensor = features_tensor.to(device)
    
    # Get the model outputs.
    with torch.no_grad():
        log_probs = model(features_tensor)  # shape: (1, 2)
        probs = torch.exp(log_probs)
    
    predicted_class = torch.argmax(probs, dim=1).item()
    probability_output = probs[0, 1].item()  # probability for class 1
    binary_output = (predicted_class == 1)
    
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

    processed_signal = preprocess_ecg(signal, fields, target_fs=400, target_length=4000)

    # TO-DO: Update to compute per-lead features. Check lead order and update and use functions for reordering leads as needed.

    # num_finite_samples = np.size(np.isfinite(signal))
    # if num_finite_samples > 0:
    #     signal_mean = np.nanmean(signal)
    # else:
    #     signal_mean = 0.0
    # if num_finite_samples > 1:
    #     signal_std = np.nanstd(signal)
    # else:
    #     signal_std = 0.0

    # features = np.concatenate(([age], one_hot_encoding_sex, [signal_mean, signal_std]))

    # return np.asarray(features, dtype=np.float32)

    return np.asarray(processed_signal, dtype=np.float32)

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
        # Input: (batch, 12, 4000)
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        
        # Fully connected layer: flattened dimension = 32*1000; output 2 classes.
        self.fc = nn.Linear(32 * 1000, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # x: (batch, 12, 4000)
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))  # -> (batch, 16, 2000)
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))  # -> (batch, 32, 1000)
        x = x.view(x.size(0), -1)  # flatten: (batch, 32000)
        x = self.fc(x)           # -> (batch, 2)
        x = self.log_softmax(x)  # log probabilities, shape: (batch, 2)
        return x

class ChallengeDataset(Dataset):
    def __init__(self, data_folder, verbose=False):
        self.data_folder = data_folder
        self.records = find_records(data_folder)
        self.verbose = verbose

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record_name = self.records[idx]
        record_path = os.path.join(self.data_folder, record_name)
        features = extract_features(record_path)   # shape: (4000, 12)
        # Transpose to (channels, time): (12, 4000)
        features = np.transpose(features, (1, 0))
        # Convert the boolean label to int (False -> 0, True -> 1)
        label = int(load_label(record_path))
        # Convert to torch tensors.
        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return features_tensor, label_tensor
    
def preprocess_ecg(signal, header, target_fs=400, target_length=3000,
                   expected_leads=['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],default_fs=500):
  
    # Convert the input signal to a numpy array.
    signal = np.array(signal)
    if signal.ndim != 2:
        raise ValueError("ECG signal must be a 2D array of shape (n_samples, n_channels)")
    n_samples, n_channels = signal.shape

    # Validate header signal length against actual samples.
    sig_len_in_header = header.get('sig_len')
    # if sig_len_in_header is not None and sig_len_in_header != n_samples:
    #     warnings.warn(f"Signal length in header ({sig_len_in_header}) does not match actual samples ({n_samples}). Using actual value.")

    # Get the sampling rate; if missing or invalid, use default.
    fs = header.get('fs', default_fs)
    if not (isinstance(fs, (int, float)) and fs > 0):
        # warnings.warn(f"Invalid or missing sampling rate in header. Using default fs = {default_fs} Hz.")
        fs = default_fs

    # Reorder the leads into the expected order.
    # If 'sig_name' is not provided, assume that the current order matches the expected order.
    available_leads = header.get('sig_name')
    if available_leads is None or not isinstance(available_leads, list):
        # warnings.warn("Lead names ('sig_name') missing in header. Assuming current column order is correct.")
        available_leads = expected_leads

    # Create an array to hold the reordered signal.
    reordered_signal = np.zeros((n_samples, len(expected_leads)))
    for i, lead in enumerate(expected_leads):
        if lead in available_leads:
            idx = available_leads.index(lead)
            if idx < n_channels:
                col = signal[:, idx].copy()
                # Handle possible missing data (NaNs) in the lead.
                if np.isnan(col).any():
                    nan_idx = np.isnan(col)
                    not_nan_idx = ~nan_idx
                    if not np.any(not_nan_idx):
                        # warnings.warn(f"All values for lead {lead} are NaN. Filling with zeros.")
                        col = np.zeros_like(col)
                    else:
                        col[nan_idx] = np.interp(np.flatnonzero(nan_idx),
                                                 np.flatnonzero(not_nan_idx),
                                                 col[not_nan_idx])
                reordered_signal[:, i] = col
        #     else:
        #         warnings.warn(f"Lead {lead} is listed in header but not present in the signal data. Filling with zeros.")
        # else:
        #     warnings.warn(f"Expected lead {lead} not found in header. Filling with zeros.")

    # Warn if any lead is entirely zeros.
    # for i, lead in enumerate(expected_leads):
    #     if np.all(reordered_signal[:, i] == 0):
    #         warnings.warn(f"Lead {lead} appears to be all zeros after reordering.")

    # Resample the signal to the target sampling rate.
    duration_sec = n_samples / fs
    resample_samples = int(round(duration_sec * target_fs))
    resampled_signal = scipy.signal.resample(reordered_signal, resample_samples, axis=0)

    # Standardize the length of the signal by padding with zeros or cropping.
    current_length = resampled_signal.shape[0]
    if current_length < target_length:
        pad_width = target_length - current_length
        final_signal = np.pad(resampled_signal, ((0, pad_width), (0, 0)), mode='constant')
    elif current_length > target_length:
        start = (current_length - target_length) // 2
        final_signal = resampled_signal[start:start + target_length, :]
    else:
        final_signal = resampled_signal

    return final_signal