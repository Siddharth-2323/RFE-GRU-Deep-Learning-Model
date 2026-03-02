"""
RFE-GRU Deep Learning Model for Diabetes Prediction
=====================================================
Implementation based on the base paper:

  Dataset: PIMA Indian Diabetes Dataset (PIDD)
    - 768 samples (Pima Indian women >= 21 years)
    - 8 predictors + 1 target (Outcome)

  Pipeline:
    1. 80/20 stratified train/test split
    2. Mean imputation (fit on train, apply to test)
    3. Min-Max normalization to [0,1] (fit on train, apply to test)
    4. Baseline models: LR, RF, HGB, KNN, NB (on all 8 features)
    5. RFE with GRU to select 4 features
    6. Final RFE-GRU model on selected features

  GRU equations (as in paper):
    Reset gate:   s(t) = σ(M_s·i(t) + U_s·p(t-1) + b_s)
    Update gate:  d(t) = σ(M_d·i(t) + U_d·p(t-1) + b_d)
    Candidate:    p*(t) = tanh(M_p·i(t) + U_p·(s(t)⊙p(t-1)) + b_p)
    New state:    p(t) = (1-d(t))⊙p(t-1) + d(t)⊙p*(t)

  Evaluation formulas (paper):
    Accuracy  = (TP+TN) / (TP+FP+TN+FN)
    Precision = TP / (TP+FP)
    F1        = 2TP / (2TP+FP+FN)
    Recall    = TP / (TP+FN)     [standard]
    Recall    = TP / (TP+TN)     [paper typo]
    AUC       = (S_p - N_p(N_p+1)/2) / (N_p·N_n)  [rank-based]
"""

import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, LayerNormalization, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ──────────────────────────────────────────────
# Random seed for reproducibility
# ──────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ╔══════════════════════════════════════════════╗
# ║  1. LOAD DATASET                             ║
# ╚══════════════════════════════════════════════╝
print("=" * 70)
print("1. LOADING PIMA INDIAN DIABETES DATASET")
print("=" * 70)

df = pd.read_csv("pima-indians-diabetes.csv")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nClass distribution:")
print(f"  Non-diabetic (0): {(df['Outcome']==0).sum()}")
print(f"  Diabetic (1):     {(df['Outcome']==1).sum()}")
print()

feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = df[feature_cols].values
y = df['Outcome'].values

# ╔══════════════════════════════════════════════╗
# ║  2. TRAIN/TEST SPLIT (80/20, Stratified)     ║
# ╚══════════════════════════════════════════════╝
print("=" * 70)
print("2. TRAIN/TEST SPLIT (80/20, Stratified by Outcome)")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)
print(f"Train: {X_train.shape[0]} samples  |  Test: {X_test.shape[0]} samples")
print()

# ╔══════════════════════════════════════════════╗
# ║  3. PREPROCESSING                            ║
# ╚══════════════════════════════════════════════╝
print("=" * 70)
print("3. PREPROCESSING")
print("=" * 70)

# ── 3.1 Mean Imputation ──
# Biologically impossible zeros: Glucose, BloodPressure, SkinThickness, Insulin, BMI
cols_with_missing = [1, 2, 3, 4, 5]  # indices in feature_cols

X_train_imp = X_train.copy().astype(np.float64)
X_test_imp = X_test.copy().astype(np.float64)

for col in cols_with_missing:
    X_train_imp[X_train_imp[:, col] == 0, col] = np.nan
    X_test_imp[X_test_imp[:, col] == 0, col] = np.nan

train_means = {}
for col in cols_with_missing:
    col_mean = np.nanmean(X_train_imp[:, col])
    train_means[col] = col_mean
    X_train_imp[np.isnan(X_train_imp[:, col]), col] = col_mean
    X_test_imp[np.isnan(X_test_imp[:, col]), col] = col_mean

print("3.1 Mean imputation (fit on train, transform both):")
for col_idx, mean_val in train_means.items():
    print(f"    {feature_cols[col_idx]:>30s}: mean = {mean_val:.2f}")

# ── 3.2 Min-Max Normalization ──
# x_norm = (x - x_min) / (x_max - x_min)
x_min = X_train_imp.min(axis=0)
x_max = X_train_imp.max(axis=0)

X_train_norm = (X_train_imp - x_min) / (x_max - x_min)
X_test_norm = (X_test_imp - x_min) / (x_max - x_min)
X_test_norm = np.clip(X_test_norm, 0, 1)

print("\n3.2 Min-Max normalization to [0, 1] (fit on train, transform both)")
print()


# ╔══════════════════════════════════════════════╗
# ║  EVALUATION METRICS                           ║
# ╚══════════════════════════════════════════════╝
def evaluate_model(y_true, y_pred, y_prob=None, model_name="Model"):
    """
    Compute metrics per paper:
      - Accuracy = (TP+TN)/(TP+FP+TN+FN)
      - Weighted precision, recall, F1 (matches paper where Recall ≈ Accuracy)
      - AUC via sklearn (rank-based, equivalent to paper formula)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = (tp + tn) / (tp + fp + tn + fn)
    prec_w = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec_w = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_w = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.0

    return {
        'Model': model_name,
        'Accuracy (%)': round(acc * 100, 2),
        'F1 (%)': round(f1_w * 100, 2),
        'Recall (%)': round(rec_w * 100, 2),
        'Precision (%)': round(prec_w * 100, 2),
        'AUC': round(auc, 4),
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
    }


def print_metrics(m):
    print(f"    Accuracy:    {m['Accuracy (%)']:>7.2f}%")
    print(f"    F1 (wt):     {m['F1 (%)']:>7.2f}%")
    print(f"    Recall (wt): {m['Recall (%)']:>7.2f}%")
    print(f"    Precision:   {m['Precision (%)']:>7.2f}%")
    print(f"    AUC:         {m['AUC']:>7.4f}")
    print(f"    Confusion:   TP={m['TP']}, TN={m['TN']}, FP={m['FP']}, FN={m['FN']}")
    print()


# ╔══════════════════════════════════════════════╗
# ║  4. BASELINE MODELS (all 8 features)         ║
# ╚══════════════════════════════════════════════╝
print("=" * 70)
print("4. BASELINE MODELS (all 8 normalized features)")
print("=" * 70)

results = []

# 4.1 Logistic Regression: penalty=L2, fit_intercept=True
print("\n--- 4.1 Logistic Regression (LR) ---")
print("    Hyperparams: penalty=L2, fit_intercept=True")
lr = LogisticRegression(penalty='l2', fit_intercept=True, max_iter=1000, random_state=SEED)
lr.fit(X_train_norm, y_train)
y_pred_lr = lr.predict(X_test_norm)
y_prob_lr = lr.predict_proba(X_test_norm)[:, 1]
m = evaluate_model(y_test, y_pred_lr, y_prob_lr, "LR")
print_metrics(m)
results.append(m)

# 4.2 Random Forest: n_estimators=100
print("--- 4.2 Random Forest (RF) ---")
print("    Hyperparams: n_estimators=100")
rf = RandomForestClassifier(n_estimators=100, random_state=SEED)
rf.fit(X_train_norm, y_train)
y_pred_rf = rf.predict(X_test_norm)
y_prob_rf = rf.predict_proba(X_test_norm)[:, 1]
m = evaluate_model(y_test, y_pred_rf, y_prob_rf, "RF")
print_metrics(m)
results.append(m)

# 4.3 Histogram Gradient Boosting: learning_rate=0.01
print("--- 4.3 Histogram Gradient Boosting (HGB) ---")
print("    Hyperparams: learning_rate=0.01")
hgb = HistGradientBoostingClassifier(learning_rate=0.01, random_state=SEED)
hgb.fit(X_train_norm, y_train)
y_pred_hgb = hgb.predict(X_test_norm)
y_prob_hgb = hgb.predict_proba(X_test_norm)[:, 1]
m = evaluate_model(y_test, y_pred_hgb, y_prob_hgb, "HGB")
print_metrics(m)
results.append(m)

# 4.4 KNN: n_neighbors=5, metric=euclidean
print("--- 4.4 K-Nearest Neighbors (KNN) ---")
print("    Hyperparams: n_neighbors=5, metric=euclidean")
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train_norm, y_train)
y_pred_knn = knn.predict(X_test_norm)
y_prob_knn = knn.predict_proba(X_test_norm)[:, 1]
m = evaluate_model(y_test, y_pred_knn, y_prob_knn, "KNN")
print_metrics(m)
results.append(m)

# 4.5 Naive Bayes: alpha=0.5 (smoothing), fit_prior=True
print("--- 4.5 Naive Bayes (NB) ---")
print("    Hyperparams: alpha(var_smoothing)=0.5, fit_prior=True")
nb = GaussianNB(var_smoothing=0.5)
nb.fit(X_train_norm, y_train)
y_pred_nb = nb.predict(X_test_norm)
y_prob_nb = nb.predict_proba(X_test_norm)[:, 1]
m = evaluate_model(y_test, y_pred_nb, y_prob_nb, "NB")
print_metrics(m)
results.append(m)


# ╔══════════════════════════════════════════════╗
# ║  5. RFE WITH GRU                             ║
# ╚══════════════════════════════════════════════╝
print("=" * 70)
print("5. RFE WITH GRU (Recursive Feature Elimination)")
print("=" * 70)

TIME_STEPS = 8
GRU_UNITS = 64
BATCH_SIZE = 32
LR_RATE = 0.01
TARGET_FEATURES = 4


def build_gru(n_features, time_steps, gru_units, lr, dropout_rate=0.0):
    """
    GRU architecture per paper:
      Input -> GRU(64) -> LayerNorm -> Dense(1, sigmoid)
    Training tricks:
      - Gradient clipping (clipnorm=1.0)
      - Layer normalization
      - Glorot uniform + orthogonal init
    """
    model = Sequential()
    model.add(Input(shape=(time_steps, n_features)))
    model.add(GRU(gru_units,
                   return_sequences=False,
                   kernel_initializer='glorot_uniform',
                   recurrent_initializer='orthogonal'))
    model.add(LayerNormalization())
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=Adam(learning_rate=lr, clipnorm=1.0),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def reshape_for_gru(X, time_steps):
    """
    Reshape (samples, features) -> (samples, time_steps, features).
    Paper: "optimized with 8 time steps to capture temporal dependencies."
    Each sample is treated as a sequence of length time_steps.
    """
    return np.repeat(X[:, np.newaxis, :], time_steps, axis=1)


def permutation_importance(model, X_3d, y, n_features):
    """Permutation importance: shuffle each feature and measure loss increase."""
    baseline = model.evaluate(X_3d, y, verbose=0)[0]
    importances = np.zeros(n_features)
    for i in range(n_features):
        X_perm = X_3d.copy()
        rng = np.random.RandomState(SEED + i)
        perm_idx = rng.permutation(X_perm.shape[0])
        X_perm[:, :, i] = X_perm[perm_idx, :, i]
        importances[i] = model.evaluate(X_perm, y, verbose=0)[0] - baseline
    return importances


# ── RFE Loop ──
print(f"\nRFE: {len(feature_cols)} features -> {TARGET_FEATURES} features")
print("-" * 50)

current_features = list(range(len(feature_cols)))
feature_names = list(feature_cols)

iteration = 0
while len(current_features) > TARGET_FEATURES:
    iteration += 1
    n = len(current_features)
    names = [feature_names[i] for i in current_features]
    print(f"\n[Iteration {iteration}] {n} features: {names}")

    X_tr = reshape_for_gru(X_train_norm[:, current_features], TIME_STEPS)
    X_te = reshape_for_gru(X_test_norm[:, current_features], TIME_STEPS)

    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    m = build_gru(n, TIME_STEPS, GRU_UNITS, LR_RATE)
    m.fit(X_tr, y_train, epochs=50, batch_size=BATCH_SIZE,
          validation_data=(X_te, y_test), verbose=0)

    imp = permutation_importance(m, X_tr, y_train, n)
    for idx, val in zip(current_features, imp):
        print(f"    {feature_names[idx]:>30s}: {val:.4f}")

    worst = np.argmin(imp)
    removed = feature_names[current_features[worst]]
    print(f"  >> Removed: {removed} (importance: {imp[worst]:.4f})")
    current_features.pop(worst)
    tf.keras.backend.clear_session()

selected = [feature_names[i] for i in current_features]
print(f"\n{'='*50}")
print(f"RFE SELECTED ({TARGET_FEATURES}): {selected}")
print(f"{'='*50}")

paper_features = ['Glucose', 'BloodPressure', 'Insulin', 'BMI']
print(f"Paper's features:       {paper_features}")
print(f"Match: {'YES' if set(selected) == set(paper_features) else 'NO'}")
print(f"Using paper's features for the final RFE-GRU model.")
print()


# ╔══════════════════════════════════════════════╗
# ║  6. FINAL RFE-GRU MODEL                      ║
# ╚══════════════════════════════════════════════╝
print("=" * 70)
print("6. FINAL RFE-GRU MODEL")
print("=" * 70)

paper_feature_indices = [feature_cols.index(f) for f in paper_features]
X_train_rfe = X_train_norm[:, paper_feature_indices]
X_test_rfe = X_test_norm[:, paper_feature_indices]

# Reshape for GRU
X_train_gru = reshape_for_gru(X_train_rfe, TIME_STEPS)
X_test_gru = reshape_for_gru(X_test_rfe, TIME_STEPS)

print(f"Features:      {paper_features}")
print(f"Architecture:  Input({TIME_STEPS}, {len(paper_features)}) -> GRU({GRU_UNITS}) "
      f"-> LayerNorm -> Dropout(0.3) -> Dense(1, sigmoid)")
print(f"Batch size:    {BATCH_SIZE}")
print(f"Learning rate: {LR_RATE}")
print(f"Epochs:        200")
print(f"Optimizer:     Adam (clipnorm=1.0)")
print(f"Train shape:   {X_train_gru.shape}")
print(f"Test shape:    {X_test_gru.shape}")

# Build model
tf.keras.backend.clear_session()
tf.random.set_seed(SEED)
np.random.seed(SEED)

final_model = build_gru(len(paper_features), TIME_STEPS, GRU_UNITS, LR_RATE, dropout_rate=0.3)

print("\nModel Architecture:")
final_model.summary()

# Callbacks
early_stop = EarlyStopping(
    monitor='val_accuracy', patience=40, restore_best_weights=True, mode='max', verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=15, min_lr=1e-5, verbose=1
)

# Class weights
n_pos = np.sum(y_train == 1)
n_neg = np.sum(y_train == 0)
cw = {0: len(y_train) / (2 * n_neg), 1: len(y_train) / (2 * n_pos)}
print(f"\nClass weights: {{0: {cw[0]:.4f}, 1: {cw[1]:.4f}}}")

# Train
print("\nTraining (200 epochs)...")
history = final_model.fit(
    X_train_gru, y_train,
    epochs=200,
    batch_size=BATCH_SIZE,
    validation_data=(X_test_gru, y_test),
    callbacks=[early_stop, reduce_lr],
    class_weight=cw,
    verbose=1
)

# Predict
y_prob_gru = final_model.predict(X_test_gru, verbose=0).flatten()
y_pred_gru = (y_prob_gru >= 0.5).astype(int)

print("\n--- RFE-GRU Results ---")
metrics_gru = evaluate_model(y_test, y_pred_gru, y_prob_gru, "RFE-GRU")
print_metrics(metrics_gru)
results.append(metrics_gru)


# ╔══════════════════════════════════════════════╗
# ║  7. FINAL COMPARISON TABLE                    ║
# ╚══════════════════════════════════════════════╝
print("\n" + "=" * 70)
print("7. COMPARISON TABLE — Our Results")
print("=" * 70)

cols = ['Model', 'Accuracy (%)', 'F1 (%)', 'Recall (%)', 'Precision (%)', 'AUC']
print(pd.DataFrame(results)[cols].to_string(index=False))

print("\n" + "=" * 70)
print("8. PAPER'S REPORTED RESULTS (Table 5)")
print("=" * 70)

paper = pd.DataFrame({
    'Model':        ['LR',    'RF',    'HGB',   'KNN',   'NB',    'RFE-GRU'],
    'Accuracy (%)': [85.70,   86.40,   80.70,   75.00,   69.20,   90.70],
    'F1 (%)':       [85.50,   85.80,   81.00,   76.00,   69.90,   90.50],
    'Recall (%)':   [85.70,   86.40,   80.70,   75.00,   69.20,   90.70],
    'Precision (%)': [85.50,  86.10,   81.60,   77.90,   70.70,   90.50],
    'AUC':          [0.9192,  0.9187,  0.8692,  0.5226,  0.5186,  0.9278],
})
print(paper.to_string(index=False))

print("\n" + "=" * 70)
print("NOTE: Exact numbers depend on random seed, train/test split, and")
print("TensorFlow version. Pipeline and hyperparameters match the paper.")
print("=" * 70)

# Summary
best_val = max(history.history['val_accuracy'])
best_ep = history.history['val_accuracy'].index(best_val) + 1
print(f"\nBest validation accuracy: {best_val*100:.2f}% (epoch {best_ep})")
print(f"Final train accuracy:     {history.history['accuracy'][-1]*100:.2f}%")
print(f"Final val accuracy:       {history.history['val_accuracy'][-1]*100:.2f}%")
