# Human Activity Recognition Project

This repository contains code and data for a Human Activity Recognition (HAR) project using the UCI HAR Dataset.

## Directory Structure

- **data/**
  - **UCI_HAR_Dataset/**
    - **train/**
      - `activity_labels.txt`
      - `features_info.txt`
      - `features.txt`
      - `README.txt`
    - **UCI HAR Dataset.names**
  - **logs/fit/**
    - `20250702-163538`
    - `20250702-164301`
    - `20250702-164742`
    - `20250702-181708`

- **model/**
  - **best_model/**
  - **har_tuner/**
  - `baseline_model.keras`
  - **notebook/**
    - `ntbk_baseline_model.ipynb`
    - `ntbk_data_preprocessing.ipynb`
    - `ntbk_eda.ipynb`
    - `ntbk_tuner_model.ipynb` (modified)
  - `README.md`

## Overview

This project aims to classify human physical activities using data from the UCI HAR Dataset. The dataset includes various features extracted from sensor data, and the project implements machine learning models to predict activities.

## Files and Folders

- **data/**: Contains the UCI HAR Dataset and related files.
  - `activity_labels.txt`: Labels for different activities.
  - `features_info.txt`: Information about the features used.
  - `features.txt`: List of features.
  - `README.txt`: Dataset documentation.
  - `UCI HAR Dataset.names`: Additional dataset metadata.
  - **logs/fit/**: Log files from model training sessions.

- **model/**: Contains trained models and notebooks.
  - **best_model/**: Directory for the best performing model.
  - **har_tuner/**: Directory for hyperparameter tuning results.
  - `baseline_model.keras`: Saved baseline model in Keras format.
  - **notebook/**: Jupyter notebooks for data preprocessing, exploratory data analysis (EDA), baseline model, and tuner model.

## Usage

1. Explore the data in the `data/UCI_HAR_Dataset` directory.
2. Review the notebooks in `model/notebook` for data preprocessing, EDA, and model training.
3. Check the logs in `model/logs/fit` for training session details.
4. Use the `best_model` directory for the final model.

## Notes

- The `ntbk_tuner_model.ipynb` file has been modified recently.
- Ensure all dependencies are installed before running the notebooks.