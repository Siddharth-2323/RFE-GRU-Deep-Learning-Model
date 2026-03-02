import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform

# 1. DATA ACQUISITION & PREPROCESSING (Page 5)
# Using standard PIMA Indian Diabetes Dataset (PIDD)
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = pd.read_csv('diabetes.csv', names=names)

# Mean Imputation for missing values (biological zeros)
cols_fix = ['plas', 'pres', 'skin', 'test', 'mass']
df[cols_fix] = df[cols_fix].replace(0, np.nan)
df.fillna(df.mean(), inplace=True)

# Min-Max Normalization to range (0, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(df.drop('class', axis=1))
y = df['class'].values

# 2. RECURSIVE FEATURE ELIMINATION (RFE) (Algorithm 1)
# Paper selection: Glucose, BloodPressure, Insulin, and BMI
# We use RF wrapper as specified in the paper's methodology
estimator = RandomForestClassifier(n_estimators=100, random_state=42)
selector = RFE(estimator, n_features_to_select=4, step=1)
X_selected = selector.fit_transform(X_scaled, y)

# 3. TEMPORAL RESHAPING (Page 10)
# Even though PIDD is static, the paper uses 8 time steps
# We reshape to: (Samples, Time Steps=8, Features=4)
X_reshaped = np.array([np.tile(row, (8, 1)) for row in X_selected])

# 4. DATA SPLIT (80% Training, 20% Testing - Page 12)
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y, test_size=0.20, random_state=42, stratify=y
)

# 5. RFE-GRU HYBRID ARCHITECTURE (Algorithm 1 & Page 15)
model = Sequential([
    Input(shape=(8, 4)), # 8 time steps, 4 features
    GRU(64, 
        kernel_initializer=GlorotUniform(seed=42), 
        activation='tanh', 
        recurrent_activation='sigmoid', 
        return_sequences=False),
    LayerNormalization(), # Prevents vanishing/exploding gradients (Page 10)
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid') # Sigmoid for probabilistic classification
])

# 6. HYPERPARAMETERS & COMPILATION
# Learning Rate: 0.01 | Optimizer: Adam (Page 15)
optimizer = Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# 7. TRAINING (200 Epochs, Batch Size 32)
model.fit(
    X_train, y_train, 
    validation_data=(X_test, y_test), 
    epochs=200, 
    batch_size=32, 
    verbose=1
)

# 8. EVALUATION PER PAPER METRICS (Eq. 10-14)
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

print("\n--- FINAL EXPERIMENTAL RESULTS ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred)*100:.2f}% (Target: 90.7%)")
print(f"Precision: {precision_score(y_test, y_pred)*100:.2f}% (Target: 90.5%)")
print(f"Recall:    {recall_score(y_test, y_pred)*100:.2f}% (Target: 90.7%)")
print(f"F1 Score:  {f1_score(y_test, y_pred)*100:.2f}% (Target: 90.5%)")
print(f"AUC:       {roc_auc_score(y_test, y_pred_prob):.4f} (Target: 0.9278)")