import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPool1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
import seaborn as sns

# dataset path 
path_mit_test = r'F:\DS-B3\MLinMedicine\practice 1\mitbih_test.csv'
path_mit_train = r'F:\DS-B3\MLinMedicine\practice 1\mitbih_train.csv'
path_ptb_norm = r'F:\DS-B3\MLinMedicine\practice 1\ptbdb_normal.csv'
path_ptb_abnorm = r'F:\DS-B3\MLinMedicine\practice 1\ptbdb_abnormal.csv'

# 1. 

# 1.1 MIT-BIH Dataset
print("\n1. MIT-BIH ARRHYTHMIA DATASET")
df_mit_train = pd.read_csv(path_mit_train, header=None)
df_mit_test = pd.read_csv(path_mit_test, header=None)

# class distribution 
labels_mit = df_mit_train.iloc[:, 187].values
unique, counts = np.unique(labels_mit, return_counts=True)
class_names = ['N (Normal)', 'S', 'V', 'F', 'Q']

print(f"\nTrain set: {len(df_mit_train)}")
print(f"Test set: {len(df_mit_test)}")
print(f"Class distribution (train set):")
for i, (cls, count) in enumerate(zip(unique, counts)):
    print(f"  {class_names[i]}: {count} samples ({count/len(df_mit_train)*100:.1f}%)")

# sample ECG signals 
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
axes = axes.ravel()
for i in range(5):
    sample = df_mit_train[df_mit_train.iloc[:, 187] == i].iloc[0, :187].values
    axes[i].plot(sample)
    axes[i].set_title(f'Class {i}: {class_names[i]}')
    axes[i].set_xlabel('Time step')
axes[5].axis('off')
plt.suptitle('MIT-BIH: ECG Signal Samples')
plt.tight_layout()
plt.savefig('mitbih_samples.png', dpi=150)
plt.show()

# 1.2 PTBDB Dataset
print("\n2. PTB DIAGNOSTIC DATASET")
df_norm = pd.read_csv(path_ptb_norm, header=None)
df_abnorm = pd.read_csv(path_ptb_abnorm, header=None)

print(f"\nNormal samples: {len(df_norm)}")
print(f"Abnormal samples: {len(df_abnorm)}")
print(f"Total: {len(df_norm) + len(df_abnorm)}")

# PTBDB samples
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(df_norm.iloc[0, :187].values)
ax1.set_title('Normal ECG')
ax1.set_xlabel('Time step')
ax2.plot(df_abnorm.iloc[0, :187].values)
ax2.set_title('Abnormal ECG')
ax2.set_xlabel('Time step')
plt.suptitle('PTBDB: ECG Signal Samples')
plt.tight_layout()
plt.savefig('ptbdb_samples.png', dpi=150)
plt.show()

# 2. Build model  

def create_model(num_classes):
    model = Sequential([
        Conv1D(64, 6, activation='relu', input_shape=(187, 1)),
        BatchNormalization(),
        MaxPool1D(3, 2),
        Conv1D(64, 3, activation='relu'),
        BatchNormalization(),
        MaxPool1D(2, 2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
    return model

# 3. Train MIT-BIH 

print("\n" + "="*60)
print("TRAINING MIT-BIH MODEL (5 classes)")
print("="*60)

# Prepare the data 
X_mit_train = df_mit_train.iloc[:, :187].values.reshape(-1, 187, 1)
y_mit_train = to_categorical(df_mit_train.iloc[:, 187])
X_mit_test = df_mit_test.iloc[:, :187].values.reshape(-1, 187, 1)
y_mit_test = to_categorical(df_mit_test.iloc[:, 187])

# Train
model_mit = create_model(5)
history_mit = model_mit.fit(
    X_mit_train, y_mit_train,
    epochs=5,
    batch_size=64,
    validation_data=(X_mit_test, y_mit_test),
    verbose=1
)

# Results 
train_acc = history_mit.history['accuracy'][-1] * 100
val_acc = history_mit.history['val_accuracy'][-1] * 100
print(f"\nMIT-BIH Results:")
print(f"Training Accuracy: {train_acc:.2f}%")
print(f"Validation Accuracy: {val_acc:.2f}%")

# training plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history_mit.history['loss'], label='Train')
ax1.plot(history_mit.history['val_loss'], label='Validation')
ax1.set_title('MIT-BIH: Loss')
ax1.set_xlabel('Epoch')
ax1.legend()
ax2.plot(history_mit.history['accuracy'], label='Train')
ax2.plot(history_mit.history['val_accuracy'], label='Validation')
ax2.set_title('MIT-BIH: Accuracy')
ax2.set_xlabel('Epoch')
ax2.legend()
plt.tight_layout()
plt.savefig('mitbih_training.png', dpi=150)
plt.show()

# Confusion Matrix
y_pred = model_mit.predict(X_mit_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_mit_test, axis=1)

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('MIT-BIH: Confusion Matrix')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('mitbih_cm.png', dpi=150)
plt.show()

# 4. Train PTBDB 

print("\n" + "="*60)
print("TRAINING PTBDB MODEL (2 classes)")
print("="*60)

# prepare the data 
df_ptb = pd.concat([df_norm, df_abnorm], ignore_index=True)
X_ptb = df_ptb.iloc[:, :187].values.reshape(-1, 187, 1)
y_ptb = to_categorical(df_ptb.iloc[:, 187])

# Train-test split
X_ptb_train, X_ptb_test, y_ptb_train, y_ptb_test = train_test_split(
    X_ptb, y_ptb, test_size=0.2, random_state=42, stratify=y_ptb
)

# Train
model_ptb = create_model(2)
history_ptb = model_ptb.fit(
    X_ptb_train, y_ptb_train,
    epochs=5,
    batch_size=64,
    validation_data=(X_ptb_test, y_ptb_test),
    verbose=1
)

# Results 
train_acc_ptb = history_ptb.history['accuracy'][-1] * 100
val_acc_ptb = history_ptb.history['val_accuracy'][-1] * 100
print(f"\nPTBDB Results:")
print(f"Training Accuracy: {train_acc_ptb:.2f}%")
print(f"Validation Accuracy: {val_acc_ptb:.2f}%")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history_ptb.history['loss'], label='Train')
ax1.plot(history_ptb.history['val_loss'], label='Validation')
ax1.set_title('PTBDB: Loss')
ax1.set_xlabel('Epoch')
ax1.legend()
ax2.plot(history_ptb.history['accuracy'], label='Train')
ax2.plot(history_ptb.history['val_accuracy'], label='Validation')
ax2.set_title('PTBDB: Accuracy')
ax2.set_xlabel('Epoch')
ax2.legend()
plt.tight_layout()
plt.savefig('ptbdb_training.png', dpi=150)
plt.show()

# Confusion Matrix PTBDB
y_pred_ptb = model_ptb.predict(X_ptb_test)
y_pred_classes_ptb = np.argmax(y_pred_ptb, axis=1)
y_true_ptb = np.argmax(y_ptb_test, axis=1)

plt.figure(figsize=(6, 5))
cm_ptb = confusion_matrix(y_true_ptb, y_pred_classes_ptb)
sns.heatmap(cm_ptb, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Abnormal'], 
            yticklabels=['Normal', 'Abnormal'])
plt.title('PTBDB: Confusion Matrix')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('ptbdb_cm.png', dpi=150)
plt.show()

# saving result 
with open('results.txt', 'w') as f:
    f.write(f"MIT-BIH Accuracy: {val_acc:.2f}%\n")
    f.write(f"PTBDB Accuracy: {val_acc_ptb:.2f}%\n")

print("\nResult saved in results.txt")
print("Plot saved in: mitbih_samples.png, ptbdb_samples.png,")
print("mitbih_training.png, ptbdb_training.png,")
print("mitbih_cm.png, ptbdb_cm.png")