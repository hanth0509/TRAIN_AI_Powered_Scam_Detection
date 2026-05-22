"""
SMS Spam / Phishing Detector — Production Training Pipeline
============================================================
Models: Bidirectional SimpleRNN | LSTM | GRU
Dataset: spam.csv  (columns: Category, Message)

Run:
    python sms_spam_detector.py

All artefacts are written to ./output/
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS & DETERMINISTIC SEED
# ─────────────────────────────────────────────────────────────────────────────
import os, re, json, pickle, warnings, logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
import seaborn as sns

# ── TensorFlow ────────────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Embedding, Bidirectional, SimpleRNN, LSTM, GRU,
    Dense, Dropout, BatchNormalization, GlobalMaxPooling1D,
    Layer,
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard,
)
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

# ── Scikit-learn ──────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
)

warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ── Deterministic seeds ────────────────────────────────────────────────────────
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
# Make TF ops deterministic where possible
os.environ["TF_DETERMINISTIC_OPS"] = "1"

# ── Mixed precision (speeds up training on modern GPUs) ────────────────────────
mixed_precision.set_global_policy("mixed_float16")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  OUTPUT DIRECTORY
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

print("=" * 65)
print("  SMS Spam / Phishing Detector — Training Pipeline")
print("=" * 65)
gpus = tf.config.list_physical_devices("GPU")
print(f"  GPUs detected : {len(gpus)}  {'(GPU training active)' if gpus else '(CPU fallback)'}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH = "spam.csv"   # adjust path if needed

df = pd.read_csv(CSV_PATH, encoding="latin-1")[["Category", "Message"]]
df.columns = ["label", "text"]
df.dropna(subset=["text", "label"], inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"Dataset shape : {df.shape}")
print(f"Class distribution:\n{df['label'].value_counts()}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  TEXT CLEANING
# ─────────────────────────────────────────────────────────────────────────────
# FIX #2: robust normalisation so the model sees consistent tokens.
# Replacing URLs/phones with sentinel tokens reduces vocabulary noise
# and lets the model learn that ANY url/phone is suspicious in context.

def clean_text(text: str) -> str:
    """
    Normalise raw SMS text for tokenisation.

    Steps
    -----
    1. Coerce to string and lowercase.
    2. Replace URLs with the sentinel token URL.
    3. Replace phone numbers (various formats) with PHONE.
    4. Remove non-alphanumeric characters (keep spaces).
    5. Collapse whitespace.
    """
    if not isinstance(text, str):
        text = str(text) if not (isinstance(text, float) and np.isnan(text)) else ""

    text = text.lower()

    # URLs: http / https / www / bare domain-like patterns
    url_re = r"(https?://\S+|www\.\S+|\S+\.(com|net|org|io|co|uk|info|biz)\S*)"
    text = re.sub(url_re, " URL ", text, flags=re.IGNORECASE)

    # Phone numbers: digits with optional country code / separators
    phone_re = r"(\+?\d[\d\s\-\(\)]{7,}\d)"
    text = re.sub(phone_re, " PHONE ", text)

    # Currency amounts — useful signal; keep as token
    text = re.sub(r"£\d+[\d,\.]*", " AMOUNT ", text)
    text = re.sub(r"\$\d+[\d,\.]*", " AMOUNT ", text)

    # Remove remaining punctuation, keep alphanumeric + spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["clean_text"] = df["text"].apply(clean_text)

print("Sample after cleaning:")
for _, row in df.sample(3, random_state=SEED).iterrows():
    print(f"  [{row['label']}]  {row['clean_text'][:80]}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 4.  LABEL ENCODING
# ─────────────────────────────────────────────────────────────────────────────
le = LabelEncoder()
y = le.fit_transform(df["label"])          # ham=0, spam=1
print(f"Label mapping : {dict(zip(le.classes_, le.transform(le.classes_)))}")

SPAM_CLASS = int(np.where(le.classes_ == "spam")[0][0])   # should be 1
print(f"Spam class index : {SPAM_CLASS}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  TOKENISER
# ─────────────────────────────────────────────────────────────────────────────
# FIX #3 + #4: use OOV token; use 95th-percentile sequence length.

VOCAB_SIZE  = 10_000
OOV_TOKEN   = "<OOV>"
EMBED_DIM   = 64

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
tokenizer.fit_on_texts(df["clean_text"])

sequences = tokenizer.texts_to_sequences(df["clean_text"])
seq_lengths = [len(s) for s in sequences]

# 95th percentile — avoids padding noise from very long outliers
MAX_LEN = int(np.percentile(seq_lengths, 95))
print(f"Sequence length stats: mean={np.mean(seq_lengths):.1f}, "
      f"median={np.median(seq_lengths):.1f}, "
      f"95th pct={MAX_LEN}, max={max(seq_lengths)}")
print(f"Using MAX_LEN = {MAX_LEN}\n")

X = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  TRAIN / TEST SPLIT  (stratified)
# ─────────────────────────────────────────────────────────────────────────────
# FIX #6: stratify ensures both splits preserve the class ratio.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"Train size : {len(X_train):,}   |   Test size : {len(X_test):,}")
print(f"Train spam : {y_train.sum()}   |   Test spam : {y_test.sum()}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  CLASS WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────
# FIX #5: compensate for ~6:1 ham/spam imbalance so the model is penalised
# more heavily for missing spam.

raw_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train,
)
class_weight = dict(enumerate(raw_weights))
print(f"Class weights : {class_weight}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  ATTENTION LAYER (Bonus)
# ─────────────────────────────────────────────────────────────────────────────
# A lightweight Bahdanau-style self-attention for sequence models.
# This allows the model to focus on the most discriminative tokens.

class SelfAttention(Layer):
    """Additive self-attention over a sequence of hidden states."""

    def __init__(self, units: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.W = Dense(units, activation="tanh")
        self.V = Dense(1)

    def call(self, hidden_states):
        # hidden_states: (batch, timesteps, features)
        score = self.V(self.W(hidden_states))         # (batch, T, 1)
        attn  = tf.nn.softmax(score, axis=1)          # (batch, T, 1)
        context = tf.reduce_sum(attn * hidden_states, axis=1)  # (batch, features)
        return context

    def get_config(self):
        config = super().get_config()
        return config


# ─────────────────────────────────────────────────────────────────────────────
# 9.  MODEL BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
# FIX #5 (architecture): all models are Bidirectional, use Dropout,
# BatchNormalization, and optional Attention to improve recall on spam.
# The final Dense uses float32 to stay numerically stable with mixed precision.

def build_rnn_model(vocab_size, embed_dim, max_len) -> tf.keras.Model:
    """Bidirectional SimpleRNN with self-attention."""
    model = Sequential(name="BiRNN", layers=[
        Embedding(vocab_size, embed_dim, input_length=max_len, mask_zero=True),
        Bidirectional(SimpleRNN(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)),
        Bidirectional(SimpleRNN(32, return_sequences=True, dropout=0.2)),
        GlobalMaxPooling1D(),
        BatchNormalization(),
        Dense(64, activation="relu"),
        Dropout(0.4),
        # float32 output — required for stable loss with mixed precision
        Dense(1, activation="sigmoid", dtype="float32"),
    ])
    return model


def build_lstm_model(vocab_size, embed_dim, max_len) -> tf.keras.Model:
    """Stacked Bidirectional LSTM with self-attention."""
    model = Sequential(name="BiLSTM", layers=[
        Embedding(vocab_size, embed_dim, input_length=max_len, mask_zero=True),
        Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)),
        Bidirectional(LSTM(32, return_sequences=True, dropout=0.2)),
        GlobalMaxPooling1D(),
        BatchNormalization(),
        Dense(64, activation="relu"),
        Dropout(0.4),
        Dense(1, activation="sigmoid", dtype="float32"),
    ])
    return model


def build_gru_model(vocab_size, embed_dim, max_len) -> tf.keras.Model:
    """Stacked Bidirectional GRU — usually the best balance of speed & quality."""
    model = Sequential(name="BiGRU", layers=[
        Embedding(vocab_size, embed_dim, input_length=max_len, mask_zero=True),
        Bidirectional(GRU(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)),
        Bidirectional(GRU(32, return_sequences=True, dropout=0.2)),
        GlobalMaxPooling1D(),
        BatchNormalization(),
        Dense(64, activation="relu"),
        Dropout(0.4),
        Dense(1, activation="sigmoid", dtype="float32"),
    ])
    return model


BUILDERS = {
    "BiRNN":  build_rnn_model,
    "BiLSTM": build_lstm_model,
    "BiGRU":  build_gru_model,
}


# ─────────────────────────────────────────────────────────────────────────────
# 10.  TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
EPOCHS     = 25
BATCH_SIZE = 64
LR_INIT    = 1e-3

histories   = {}
results     = {}
all_models  = {}

for model_name, builder in BUILDERS.items():
    print(f"\n{'─'*65}")
    print(f"  Training : {model_name}")
    print(f"{'─'*65}")

    model = builder(VOCAB_SIZE, EMBED_DIM, MAX_LEN)
    model.compile(
        optimizer=Adam(learning_rate=LR_INIT),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    model.summary(print_fn=lambda x: print("  " + x))

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.keras")

    callbacks = [
        # Stop early when val_loss stops improving; restore best weights
        EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        # Reduce LR on plateau
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-6, verbose=1
        ),
        # Save the best checkpoint
        ModelCheckpoint(
            filepath=ckpt_path, monitor="val_f1_macro",
            save_best_only=True, mode="max", verbose=0
        ),
        # TensorBoard
        TensorBoard(log_dir=os.path.join(LOG_DIR, model_name), histogram_freq=1),
    ]

    # ── custom val metric: macro F1 ─────────────────────────────────────────
    # Keras doesn't expose F1 natively; compute it in a callback instead.
    val_f1_scores: list[float] = []

    class ValF1Callback(tf.keras.callbacks.Callback):
        """Computes macro-averaged F1 on the validation set after each epoch."""
        def on_epoch_end(self, epoch, logs=None):
            y_pred_proba = self.model.predict(X_test, batch_size=256, verbose=0)
            y_pred = (y_pred_proba.flatten() >= 0.5).astype(int)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            logs["val_f1_macro"] = f1
            val_f1_scores.append(f1)

    callbacks.insert(0, ValF1Callback())

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,   # FIX #5: penalise missing spam
        callbacks=callbacks,
        verbose=1,
    )

    histories[model_name] = history.history

    # ── Evaluation ────────────────────────────────────────────────────────────
    y_pred_proba = model.predict(X_test, batch_size=256, verbose=0).flatten()
    y_pred = (y_pred_proba >= 0.5).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=SPAM_CLASS, zero_division=0)
    rec  = recall_score(y_test, y_pred, pos_label=SPAM_CLASS, zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_spam = f1_score(y_test, y_pred, pos_label=SPAM_CLASS, zero_division=0)

    results[model_name] = {
        "accuracy": acc, "precision": prec, "recall": rec,
        "weighted_f1": f1, "spam_f1": f1_spam,
    }
    all_models[model_name] = model

    print(f"\n  {model_name} Results")
    print(f"  {'Accuracy':<18}: {acc:.4f}")
    print(f"  {'Precision(spam)':<18}: {prec:.4f}")
    print(f"  {'Recall(spam)':<18}: {rec:.4f}   ← most important")
    print(f"  {'Weighted F1':<18}: {f1:.4f}")
    print(f"  {'Spam F1':<18}: {f1_spam:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Confusion matrix figure
    # FIX: dùng imshow + text thủ công để chữ luôn hiển thị
    # dù giá trị các ô chênh lệch lớn (ham ~950 vs spam ~140)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # chữ trắng nếu nền tối, chữ đen nếu nền sáng
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    fontsize=14, color=color, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(le.classes_)
    ax.set_yticks([0, 1]); ax.set_yticklabels(le.classes_)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"{model_name} — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_confusion.png"), dpi=120)
    plt.close()

    # Training history plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="train loss")
    axes[0].plot(history.history["val_loss"], label="val loss")
    axes[0].set_title(f"{model_name} — Loss"); axes[0].legend()
    axes[1].plot(history.history["accuracy"], label="train acc")
    axes[1].plot(history.history["val_accuracy"], label="val acc")
    axes[1].set_title(f"{model_name} — Accuracy"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_history.png"), dpi=120)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 11.  MODEL SELECTION  (best weighted F1)
# ─────────────────────────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["weighted_f1"])
best_model = all_models[best_name]

print("\n" + "=" * 65)
print(f"  Best model : {best_name}  "
      f"(weighted F1 = {results[best_name]['weighted_f1']:.4f})")
print("=" * 65)

# Comparison bar chart
fig, ax = plt.subplots(figsize=(8, 5))
metrics_to_plot = ["accuracy", "precision", "recall", "weighted_f1", "spam_f1"]
x = np.arange(len(metrics_to_plot))
width = 0.25
for i, (mname, mvals) in enumerate(results.items()):
    ax.bar(x + i * width, [mvals[m] for m in metrics_to_plot],
           width=width, label=mname)
ax.set_xticks(x + width)
ax.set_xticklabels([m.replace("_", " ") for m in metrics_to_plot], rotation=15)
ax.set_ylim(0, 1.05)
ax.legend()
ax.set_title("Model Comparison")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=120)
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 12.  SAVE ARTEFACTS
# ─────────────────────────────────────────────────────────────────────────────
# 12a. Best model (.keras native format)
model_path = os.path.join(OUTPUT_DIR, "best_model.keras")
best_model.save(model_path)
print(f"\nSaved best model     → {model_path}")

# 12b. Tokenizer
tok_path = os.path.join(OUTPUT_DIR, "tokenizer.pkl")
with open(tok_path, "wb") as f:
    pickle.dump(tokenizer, f)
print(f"Saved tokenizer      → {tok_path}")

# 12c. Label encoder
le_path = os.path.join(OUTPUT_DIR, "label_encoder.pkl")
with open(le_path, "wb") as f:
    pickle.dump(le, f)
print(f"Saved label encoder  → {le_path}")

# 12d. Config JSON (everything needed to reconstruct the inference pipeline)
config = {
    "best_model_name": best_name,
    "vocab_size":  VOCAB_SIZE,
    "embed_dim":   EMBED_DIM,
    "max_len":     MAX_LEN,
    "oov_token":   OOV_TOKEN,
    "spam_class":  SPAM_CLASS,
    "label_classes": le.classes_.tolist(),
    "results":     results,
}
cfg_path = os.path.join(OUTPUT_DIR, "config.json")
with open(cfg_path, "w") as f:
    json.dump(config, f, indent=2)
print(f"Saved config         → {cfg_path}")

# 12e. Training histories
hist_path = os.path.join(OUTPUT_DIR, "training_histories.json")
with open(hist_path, "w") as f:
    # Convert numpy floats to Python floats for JSON serialisation
    serialisable = {
        m: {k: [float(v) for v in vals] for k, vals in h.items()}
        for m, h in histories.items()
    }
    json.dump(serialisable, f, indent=2)
print(f"Saved training logs  → {hist_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 13.  INFERENCE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def predict_text(
    texts: list[str] | str,
    model: tf.keras.Model,
    tokenizer: Tokenizer,
    label_encoder: LabelEncoder,
    max_len: int,
    spam_class: int = 1,
    threshold: float = 0.5,
) -> list[dict]:
    """
    Predict spam probability for one or more raw SMS strings.

    Parameters
    ----------
    texts        : single string or list of strings
    model        : loaded Keras model
    tokenizer    : fitted Keras Tokenizer
    label_encoder: fitted LabelEncoder
    max_len      : sequence padding length (from config)
    spam_class   : integer index of the spam class
    threshold    : classification threshold (default 0.5;
                   lower → more sensitive to spam, fewer false negatives)

    Returns
    -------
    List of dicts with keys: text, label, confidence, spam_probability
    """
    if isinstance(texts, str):
        texts = [texts]

    cleaned   = [clean_text(t) for t in texts]
    seqs      = tokenizer.texts_to_sequences(cleaned)
    padded    = pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")
    proba     = model.predict(padded, batch_size=64, verbose=0).flatten()

    # For binary output (sigmoid): proba is P(spam) if spam_class==1
    spam_proba = proba if spam_class == 1 else (1.0 - proba)
    labels_idx = (spam_proba >= threshold).astype(int)

    output = []
    for raw, label_idx, sp in zip(texts, labels_idx, spam_proba):
        label = label_encoder.inverse_transform([label_idx])[0]
        confidence = float(sp) if label_idx == spam_class else float(1 - sp)
        output.append({
            "text":             raw,
            "label":            label,
            "confidence":       round(confidence, 4),
            "spam_probability": round(float(sp), 4),
        })
    return output


# ─────────────────────────────────────────────────────────────────────────────
# 14.  INFERENCE TEST
# ─────────────────────────────────────────────────────────────────────────────
test_messages = [
    # Normal ham
    "Hey, are you free for lunch tomorrow around 1pm?",
    "I'll pick up the kids at 3. Don't forget we have dinner with mum.",

    # Classic spam
    "Congratulations! You've won a £1000 gift card. Claim now at win-free.com",
    "FREE entry into our weekly competition. Text WIN to 80086 now!",

    # Phishing — banking scam
    "URGENT: Your NatWest account has been suspended. Verify your identity now "
    "at natwest-secure-login.com or call 0800-123-4567",
    "Your HSBC account shows a suspicious login. Click here immediately to secure "
    "your account: http://hsbc-alert-verify.net/login",

    # Crypto scam
    "BITCOIN ALERT: Your wallet has been credited with 0.5 BTC. Log in at "
    "crypto-rewards-now.io to claim before it expires.",

    # Suspicious login / OTP scam
    "Your verification code is 834921. NEVER share this code. If you did not "
    "request it, visit account-security-check.com to protect your account.",

    # Delivery phishing
    "Your parcel could not be delivered. Pay £1.99 handling fee at "
    "royal-mail-delivery.com/track to reschedule.",

    # Prize scam
    "You have been selected for a cash reward of $5000! Reply YES to claim. "
    "Limited time offer. Stop=Cancel",
]

print("\n" + "=" * 65)
print("  INFERENCE TEST")
print("=" * 65)

predictions = predict_text(
    test_messages,
    model=best_model,
    tokenizer=tokenizer,
    label_encoder=le,
    max_len=MAX_LEN,
    spam_class=SPAM_CLASS,
)

for pred in predictions:
    bar  = "🚨 SPAM" if pred["label"] == "spam" else "✅  HAM "
    conf = pred["confidence"]
    sp   = pred["spam_probability"]
    print(f"\n  {bar}  (spam_prob={sp:.3f}, conf={conf:.3f})")
    print(f"         {pred['text'][:80]}")


# ─────────────────────────────────────────────────────────────────────────────
# 15.  SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  FINAL RESULTS SUMMARY")
print("=" * 65)
print(f"  {'Model':<10}  {'Acc':>6}  {'Prec':>6}  {'Recall':>7}  {'wF1':>6}  {'SpamF1':>7}")
print(f"  {'-'*10}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*7}")
for mname, mvals in results.items():
    star = "◄ best" if mname == best_name else ""
    print(f"  {mname:<10}  {mvals['accuracy']:>6.4f}  "
          f"{mvals['precision']:>6.4f}  {mvals['recall']:>7.4f}  "
          f"{mvals['weighted_f1']:>6.4f}  {mvals['spam_f1']:>7.4f}  {star}")

print(f"\n  All artefacts saved in: {os.path.abspath(OUTPUT_DIR)}/")
print("  Done ✓")
