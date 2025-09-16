import os
from typing import List, Dict
import mne

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings 
warnings.filterwarnings("ignore")
from tqdm import tqdm
tqdm.pandas()

os.chdir("./main")

# sklearn packages 
from sklearn.model_selection import train_test_split


eeg_data = pd.read_pickle("../data/processed/eeg_processed.pkl")
eeg_data["shape"] = eeg_data["epoch_data"].apply(lambda x : x.shape)
eeg_data = eeg_data[eeg_data["shape"]!=(64, 0)]
eeg_data.head()


X = eeg_data.drop(["label", "label_description"], axis = 1) # Features 
y = eeg_data[["label", "label_description"]] # Target Column

stratify_by = eeg_data[["label_description"]]

# Perform the stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size = 0.3, 
        random_state = 42, 
        stratify= stratify_by
)
 
print(f"Shape of X_train : {X_train.shape}")
print(f"Shape of X_test : {X_test.shape}")
print(f"Shape of y_train : {y_train.shape}")
print(f"Shape of y_test : {y_test.shape}")


# Define constants once
CH_NAMES = ['Fc5.','Fc3.','Fc1.','Fcz.','Fc2.','Fc4.','Fc6.','C5..','C3..','C1..','Cz..','C2..','C4..','C6..',
            'Cp5.','Cp3.','Cp1.','Cpz.','Cp2.','Cp4.','Cp6.','Fp1.','Fpz.','Fp2.','Af7.','Af3.','Afz.','Af4.',
            'Af8.','F7..','F5..','F3..','F1..','Fz..','F2..','F4..','F6..','F8..','Ft7.','Ft8.','T7..','T8..',
            'T9..','T10.','Tp7.','Tp8.','P7..','P5..','P3..','P1..','Pz..','P2..','P4..','P6..','P8..','Po7.',
            'Po3.','Poz.','Po4.','Po8.','O1..','Oz..','O2..','Iz..']
SFREQ = 160
CH_TYPES = ["eeg"] * 64



def zscore_normalization(eeg_data:np.array) -> np.array:
    """
        EEG Data : numpy array of shape (n_channels, n_samples)
    """

    mean = np.mean(eeg_data, axis = 1, keepdims= True)
    std = np.std(eeg_data, axis = 1, keepdims=True)
    return (eeg_data - mean)/std


X_train["epoch_data_normalized"] = X_train["epoch_data"].apply(zscore_normalization)


def band_pass_filter(l_freq: float, h_freq: float, eeg_data: np.ndarray) -> np.ndarray:
    """
    Apply band-pass filter using MNE's filter_data.
    eeg_data: shape (n_channels, n_times)
    """
    return mne.filter.filter_data(
        data=eeg_data,
        sfreq=SFREQ,
        l_freq=l_freq,
        h_freq=h_freq,
        verbose=False
    )
X_train["epoch_data_bandpass"] = X_train["epoch_data_normalized"].apply(lambda x : band_pass_filter(l_freq=0.5, 
                                                                                                    h_freq=50, 
                                                                                                    eeg_data= x))


INFO = mne.create_info(ch_names=CH_NAMES, sfreq=SFREQ, ch_types=CH_TYPES)

def run_ica(eeg_data: np.ndarray, n_components: int = 64, method: str = "fastica", random_state: int = 97) -> np.ndarray:
    """
    Run ICA on EEG data (single epoch) and return artifact-cleaned EEG.
    
    Parameters:
    -----------
    eeg_data : np.ndarray
        EEG data of shape (n_channels, n_times).
    n_components : int
        Number of ICA components to estimate.
    method : str
        ICA method ('fastica', 'infomax', 'picard').
    random_state : int
        Reproducibility seed.
    
    Returns:
    --------
    np.ndarray : Cleaned EEG data of shape (n_channels, n_times).
    """
    # Create RawArray for MNE ICA
    raw = mne.io.RawArray(data=eeg_data, info=INFO, verbose=False)

    # Initialize ICA
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method=method,
        random_state=random_state,
        max_iter="auto"
    )
    
    # Fit ICA
    ica.fit(raw)

    #Automatically detect EOG artifacts if EOG channel exists
    eog_inds, _ = ica.find_bads_eog(raw, ch_name = ["Fp1.", 
                                           "Fpz.", 
                                           "Fp2."])
    ica.exclude = eog_inds

    # Apply ICA
    raw_clean = raw.copy()
    ica.apply(raw_clean)

    return raw_clean.get_data()


def run_ica_safe(eeg_data: np.ndarray, n_components: int = 20):
    try:
        # Fill NaNs first
        eeg_data = np.nan_to_num(eeg_data, 0.0)
        # Run ICA (reuse your previous function)
        return run_ica(eeg_data, n_components=n_components)
    except Exception as e:
        print(f"ICA failed for an epoch: {e}")
        # Return original EEG data if ICA fails
        return "Error"

# Apply with progress bar
results = []
for eeg_data in tqdm(X_train["epoch_data_bandpass"], desc="ICA robust"):
    results.append(run_ica_safe(eeg_data))


X_train["epoch_data_ICA"] = results

X_train.to_pickle("../data/processed/eeg_processed_X_train.pkl")