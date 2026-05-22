# Email Spam Classification - Local Training Setup

## Overview
This project trains RNN, LSTM, and GRU models for email spam classification. Originally developed on Google Colab, now adapted for local execution with virtual environment.

## Prerequisites
- Python 3.9 or higher
- pip (Python package installer)

## Setup Instructions

### 1. Create Virtual Environment
```powershell
# Navigate to project directory
cd "f:\HOAIHA\HK2 2026-2027\DACS3\TRAIN"

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
.\venv\Scripts\activate
```

### 2. Install Dependencies
```powershell
# Make sure venv is activated first
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify Dataset
Ensure `spam.csv` is in the project directory:
```
f:\HOAIHA\HK2 2026-2027\DACS3\TRAIN\spam.csv
```

### 4. Run Training
```powershell
# Make sure venv is activated
python train.py
```

## Project Structure
```
TRAIN/
├── train.py              # Main training script
├── requirements.txt      # Python dependencies
├── spam.csv             # Dataset file
├── venv/                # Virtual environment (created after setup)
└── models/              # Saved models (created after training)
    ├── sms_best_model.keras
    ├── sms_tokenizer.pkl
    ├── sms_label_encoder.pkl
    ├── model_config.json
    └── model_comparison.png
```

## Key Changes from Colab Version
- Removed Google Drive mount code
- Changed dataset path from `/content/drive/MyDrive/...` to local `spam.csv`
- Removed `!pip install` commands (use requirements.txt instead)
- Removed image display code that relied on Drive paths
- Model save path changed to local `./models/` directory
- Added automatic directory creation for model saving

## Configuration
You can modify these paths in `train.py`:
```python
DATA_PATH = 'spam.csv'              # Dataset location
MODEL_SAVE_PATH = './models/'       # Where to save trained models
```

## Deactivating Virtual Environment
```powershell
deactivate
```

## Troubleshooting
- If you get "ModuleNotFoundError", ensure venv is activated and dependencies are installed
- If dataset not found, check that `spam.csv` is in the correct directory
- For GPU support, install CUDA-compatible TensorFlow version if needed
