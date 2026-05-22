import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# ======================================================
# TENSORFLOW / KERAS
# ======================================================

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import (
    Embedding,
    SimpleRNN,
    LSTM,
    GRU,
    Dense,
    Dropout
)

from keras.callbacks import EarlyStopping

import tensorflow as tf
import keras

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")

# ======================================================
# CONFIGURATION
# ======================================================

# Local paths - adjust these to your local directory structure
DATA_PATH = 'spam.csv'
MODEL_SAVE_PATH = './models/'

# ======================================================
# DATA LOADING
# ======================================================

print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"Data shape: {df.shape}")
print(df.head())

print("\nClass distribution:")
print(df['Category'].value_counts())

print("\nData info:")
print(df.info())

print("\nData description:")
print(df.describe())

print(f"\nNumber of duplicates: {df.duplicated().sum()}")

# ======================================================
# DATA PREPROCESSING
# ======================================================

print("\nPreprocessing data...")

# Remove duplicates
df.drop_duplicates(inplace=True)
print(f"Data shape after removing duplicates: {df.shape}")

# Encode labels
encoder = LabelEncoder()
df['Category'] = encoder.fit_transform(df['Category'])
print(f"Labels encoded: {encoder.classes_}")

# Tokenize text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['Message'])
sequences = tokenizer.texts_to_sequences(df['Message'])

# Calculate max sequence length
max_len = max([len(seq) for seq in sequences])
print(f"Max sequence length: {max_len}")

# Pad sequences
X = pad_sequences(sequences, maxlen=max_len)
y = df['Category']

# Vocabulary size
vocabs = len(tokenizer.word_index) + 1
print(f"Vocabulary size: {vocabs}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42
)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# ======================================================
# MODEL ARCHITECTURES
# ======================================================

print("\n" + "="*50)
print("BUILDING MODELS")
print("="*50)

# RNN Model
print("\nBuilding RNN model...")
rnn_model = Sequential([
    Embedding(vocabs, 64, input_length=max_len),
    SimpleRNN(128),
    Dense(1, activation='sigmoid')
])
rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
rnn_model.summary()

# LSTM Model
print("\nBuilding LSTM model...")
lstm_model = Sequential([
    Embedding(vocabs, 64, input_length=max_len),
    LSTM(128),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.summary()

# GRU Model
print("\nBuilding GRU model...")
gru_model = Sequential([
    Embedding(vocabs, 64, input_length=max_len),
    GRU(128),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])
gru_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
gru_model.summary()

# ======================================================
# CALLBACKS
# ======================================================

rnn_cb = EarlyStopping(patience=5, restore_best_weights=True)
lstm_cb = EarlyStopping(patience=5, restore_best_weights=True)
gru_cb = EarlyStopping(patience=5, restore_best_weights=True)

# ======================================================
# TRAIN MODELS
# ======================================================

print("\n" + "="*50)
print("TRAINING MODELS")
print("="*50)

print("\n" + "-"*30 + " RNN " + "-"*30)
history_rnn = rnn_model.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_test, y_test),
    batch_size=16,
    callbacks=rnn_cb,
    verbose=1
)

print("\n" + "-"*30 + " LSTM " + "-"*30)
history_lstm = lstm_model.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_test, y_test),
    batch_size=16,
    callbacks=lstm_cb,
    verbose=1
)

print("\n" + "-"*30 + " GRU " + "-"*30)
history_gru = gru_model.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_test, y_test),
    batch_size=16,
    callbacks=gru_cb,
    verbose=1
)

# ======================================================
# EVALUATE MODELS
# ======================================================

print("\n" + "="*50)
print("EVALUATING MODELS")
print("="*50)

# Predictions
y_pred_rnn = rnn_model.predict(X_test)
y_pred_lstm = lstm_model.predict(X_test)
y_pred_gru = gru_model.predict(X_test)

print(f"\nRNN Model Accuracy: {accuracy_score(y_test, y_pred_rnn.round()):.4f}")
print(f"LSTM Model Accuracy: {accuracy_score(y_test, y_pred_lstm.round()):.4f}")
print(f"GRU Model Accuracy: {accuracy_score(y_test, y_pred_gru.round()):.4f}")

# Classification reports
print("\n" + "="*30 + " RNN Report " + "="*30)
print(classification_report(y_test, y_pred_rnn.round()))

print("\n" + "="*30 + " LSTM Report " + "="*30)
print(classification_report(y_test, y_pred_lstm.round()))

print("\n" + "="*30 + " GRU Report " + "="*30)
print(classification_report(y_test, y_pred_gru.round()))

# ======================================================
# MODEL COMPARISON
# ======================================================

print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)

results = []
models = {
    "RNN": rnn_model,
    "LSTM": lstm_model,
    "GRU": gru_model
}

for name, model in models.items():
    y_pred = model.predict(X_test).round()
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": report['weighted avg']['precision'],
        "Recall": report['weighted avg']['recall'],
        "F1-score": report['weighted avg']['f1-score']
    })

df_results = pd.DataFrame(results)
print(df_results)

# Create save directory if it doesn't exist
import os
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Plot comparison
df_results.set_index("Model")[["Accuracy", "F1-score"]].plot(kind="bar")
plt.title("Model Comparison")
plt.tight_layout()
plt.savefig(MODEL_SAVE_PATH + 'model_comparison.png')
plt.show()

# Select best model
best_model_row = df_results.loc[df_results['F1-score'].idxmax()]
best_model_name = best_model_row['Model']
best_model = models[best_model_name]

print(f"\nBest model: {best_model_name}")

# ======================================================
# SAVE MODELS
# ======================================================

print("\n" + "="*50)
print("SAVING MODELS")
print("="*50)

import json
import joblib

# Save full model (.keras)
best_model.save(MODEL_SAVE_PATH + 'sms_best_model.keras')
print("✅ Full model (.keras) saved!")

# Save tokenizer
joblib.dump(tokenizer, MODEL_SAVE_PATH + 'sms_tokenizer.pkl')
print("✅ Tokenizer saved!")

# Save label encoder
joblib.dump(encoder, MODEL_SAVE_PATH + 'sms_label_encoder.pkl')
print("✅ Label encoder saved!")

# Save model config
model_config = {
    "max_len": int(max_len),
    "vocab_size": int(vocabs),
    "embedding_dim": 64,
    "units": 128,
    "dropout_rate": 0.2,
    "model_type": best_model_name,
    "loss": "binary_crossentropy",
    "optimizer": "adam",
    "spam_threshold": 0.5
}

with open(MODEL_SAVE_PATH + 'model_config.json', 'w') as f:
    json.dump(model_config, f, indent=4)
print("✅ Model config saved!")

# ======================================================
# INFERENCE FUNCTION
# ======================================================

def classify_email(text, rnn_model, lstm_model, gru_model, tokenizer, max_len):
    """
    Classifies the given email text as spam or ham using RNN, LSTM, and GRU models.
    """
    # Preprocess the input text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)

    # Make predictions using each model
    rnn_prediction = rnn_model.predict(padded_sequence)[0][0]
    lstm_prediction = lstm_model.predict(padded_sequence)[0][0]
    gru_prediction = gru_model.predict(padded_sequence)[0][0]

    # Interpret the results
    results = {
        'Simple RNN Prediction': 'Spam' if rnn_prediction >= 0.5 else 'Ham',
        'LSTM Prediction': 'Spam' if lstm_prediction >= 0.5 else 'Ham',
        'GRU Prediction': 'Spam' if gru_prediction >= 0.5 else 'Ham',
        'Raw RNN Score': float(rnn_prediction),
        'Raw LSTM Score': float(lstm_prediction),
        'Raw GRU Score': float(gru_prediction)
    }

    return results

# Test inference
print("\n" + "="*50)
print("TEST INFERENCE")
print("="*50)

test_email = "Congratulations! You've won a free iPhone. Click here to claim your prize!"
classification_results = classify_email(test_email, rnn_model, lstm_model, gru_model, tokenizer, max_len)

print(f"Test Email: '{test_email}'")
print(f"Best Model ({best_model_name}) Prediction: {classification_results[f'{best_model_name} Prediction']} (Raw Score: {classification_results[f'Raw {best_model_name} Score']:.4f})")

print("\n" + "="*50)
print("TRAINING COMPLETE!")
print("="*50)
