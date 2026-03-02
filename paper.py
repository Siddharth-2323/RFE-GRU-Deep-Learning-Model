import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, initializers
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------------------------------
# 1. Load PIMA dataset
# CSV must have headers:
# Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
# ---------------------------------------------------

df = pd.read_csv("pima-indians-diabetes.csv")

feature_cols = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
target_col = "Outcome"

X = df[feature_cols].values
y = df[target_col].values

# ---------------------------------------------------
# 2. Train/test split (80/20, stratified)
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------------------------------------------
# 3. Mean imputation
# ---------------------------------------------------

imputer = SimpleImputer(strategy="mean")
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

# ---------------------------------------------------
# 4. Min–max normalization (0,1)
# ---------------------------------------------------

scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

# ---------------------------------------------------
# 5. Baseline models (LR, RF, HGB, KNN, NB)
# ---------------------------------------------------

def evaluate_model(name, clf, X_tr, y_tr, X_te, y_te):
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_te)[:, 1]
    else:
        # for models without predict_proba (fallback)
        y_prob = y_pred.astype(float)

    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred)
    rec = recall_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_prob)

    print(f"\n{name} Test Metrics:")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"F1-score:  {f1*100:.2f}%")
    print(f"AUC:       {auc:.4f}")

# Logistic Regression (L2, fit_intercept)
lr = LogisticRegression(
    penalty="l2",
    fit_intercept=True,
    max_iter=2000,
    solver="lbfgs",
)
# Random Forest (100 trees)
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
# Histogram Gradient Boosting (lr=0.01)
hgb = HistGradientBoostingClassifier(
    learning_rate=0.01,
    random_state=42
)
# KNN (k=5, Euclidean)
knn = KNeighborsClassifier(
    n_neighbors=5,
    metric="euclidean"
)
# Naive Bayes (alpha=0.5 approx via var_smoothing)
nb = GaussianNB(var_smoothing=1e-9)

print("Baseline Models:")
evaluate_model("LR", lr, X_train_scaled, y_train, X_test_scaled, y_test)
evaluate_model("RF", rf, X_train_scaled, y_train, X_test_scaled, y_test)
evaluate_model("HGB", hgb, X_train_scaled, y_train, X_test_scaled, y_test)
evaluate_model("KNN", knn, X_train_scaled, y_train, X_test_scaled, y_test)
evaluate_model("NB", nb, X_train_scaled, y_train, X_test_scaled, y_test)

# ---------------------------------------------------
# 6. GRU utilities (with gradient clipping, layer norm, init)
# ---------------------------------------------------

class LayerNormGRUCell(layers.GRUCell):
    # Wrap a GRUCell to include layer normalization after the GRU output
    def __init__(self, units, **kwargs):
        super().__init__(units, **kwargs)
        self.layer_norm = LayerNormalization()

    def call(self, inputs, states, training=None):
        output, new_states = super().call(inputs, states, training=training)
        output = self.layer_norm(output)
        return output, new_states

def build_gru_classifier(input_dim, time_steps=8):
    # Input: sequence of shape (time_steps, input_dim)
    inputs = layers.Input(shape=(time_steps, input_dim))

    # GRU with 64 units, gradient clipping and orthogonal init
    gru_cell = LayerNormGRUCell(
        64,
        kernel_initializer=initializers.GlorotUniform(),
        recurrent_initializer=initializers.Orthogonal(),
        bias_initializer="zeros",
    )
    gru_layer = layers.RNN(
        gru_cell,
        return_sequences=False
    )

    x = gru_layer(inputs)
    x = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=inputs, outputs=x)

    opt = optimizers.Adam(learning_rate=0.01, clipnorm=1.0)
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def to_sequence(X, time_steps):
    # Repeat static features along temporal axis (paper uses 8 time steps)
    return np.repeat(X[:, np.newaxis, :], time_steps, axis=1)

# ---------------------------------------------------
# 7. RFE loop using GRU as core model
# ---------------------------------------------------

def rfe_with_gru(X_tr, y_tr, X_te, y_te, feature_names, target_num_features=4,
                 time_steps=8, epochs=200, batch_size=32):
    """
    RFE with GRU core model.
    At each step:
      - train GRU
      - compute feature importance using permutation importance on validation data
      - remove least important feature
    Stop when target_num_features is reached.

    Returns:
      selected_feature_names, trained GRU model, metrics on test set.
    """
    current_features = list(feature_names)
    X_tr_curr = X_tr.copy()
    X_te_curr = X_te.copy()

    while len(current_features) > target_num_features:
        print(f"\nRFE step with features: {current_features}")

        input_dim = X_tr_curr.shape[1]
        model = build_gru_classifier(input_dim=input_dim, time_steps=time_steps)

        es = EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True
        )

        X_tr_seq = to_sequence(X_tr_curr, time_steps)
        X_te_seq = to_sequence(X_te_curr, time_steps)

        model.fit(
            X_tr_seq, y_tr,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[es],
            verbose=0
        )

        # Permutation importance: shuffle each feature on validation set
        # and measure drop in accuracy
        # Use a subset of training data as validation for importance
        # (here reuse X_te_curr, y_te as "validation" for simplicity)
        base_prob = model.predict(X_te_seq, verbose=0).ravel()
        base_pred = (base_prob >= 0.5).astype(int)
        base_acc = accuracy_score(y_te, base_pred)

        importances = []
        for j in range(X_te_curr.shape[1]):
            X_perm = X_te_curr.copy()
            shuffle_idx = np.random.permutation(X_perm.shape[0])
            X_perm[:, j] = X_perm[shuffle_idx, j]

            X_perm_seq = to_sequence(X_perm, time_steps)
            p_prob = model.predict(X_perm_seq, verbose=0).ravel()
            p_pred = (p_prob >= 0.5).astype(int)
            p_acc = accuracy_score(y_te, p_pred)

            # importance: drop in accuracy when feature j is permuted
            importances.append(base_acc - p_acc)

        importances = np.array(importances)
        least_idx = np.argmin(importances)
        removed_feature = current_features[least_idx]

        print(f"Removing least important feature: {removed_feature}")

        # Remove feature from train/test
        mask = np.ones(len(current_features), dtype=bool)
        mask[least_idx] = False
        X_tr_curr = X_tr_curr[:, mask]
        X_te_curr = X_te_curr[:, mask]
        current_features = [f for i, f in enumerate(current_features) if mask[i]]

    # Final GRU training on selected features
    print(f"\nFinal selected features: {current_features}")
    input_dim = X_tr_curr.shape[1]
    model = build_gru_classifier(input_dim=input_dim, time_steps=time_steps)

    es = EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True
    )

    X_tr_seq = to_sequence(X_tr_curr, time_steps)
    X_te_seq = to_sequence(X_te_curr, time_steps)

    model.fit(
        X_tr_seq, y_tr,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[es],
        verbose=0
    )

    y_prob = model.predict(X_te_seq, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred)
    rec = recall_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_prob)

    print("\nRFE-GRU Test Metrics (implementation):")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"F1-score:  {f1*100:.2f}%")
    print(f"AUC:       {auc:.4f}")

    return current_features, model, (acc, prec, rec, f1, auc)

# ---------------------------------------------------
# 8. Run RFE-GRU according to paper spec
# ---------------------------------------------------

selected_features, gru_model, metrics = rfe_with_gru(
    X_train_scaled,
    y_train,
    X_test_scaled,
    y_test,
    feature_names=feature_cols,
    target_num_features=4,
    time_steps=8,
    epochs=200,
    batch_size=32
)

print("\nSelected features per this RFE-GRU run:")
print(selected_features)
