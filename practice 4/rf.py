# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# %%
df = pd.read_csv('C:/DS-B3/MLinMedicine/practice 4/annotations.csv')

print(f"Total of nodes: {len(df)}")
print(f"Different CT scan images: {df['seriesuid'].nunique()}")
print("\nfirst 5 lines:")
print(df.head())

# %% [markdown]
# # 2. Creating features from existing data

# %%
# Basic features from coordinates
df['x_abs'] = np.abs(df['coordX'])
df['y_abs'] = np.abs(df['coordY'])
df['z_abs'] = np.abs(df['coordZ'])

# Distance from the origin (possibly related to location within the lungs)
df['distance_from_origin'] = np.sqrt(df['coordX']**2 + df['coordY']**2 + df['coordZ']**2)

# Correlations between coordinates 
df['xy_ratio'] = df['coordX'] / (df['coordY'] + 1e-6)  
df['xz_ratio'] = df['coordX'] / (df['coordZ'] + 1e-6)
df['yz_ratio'] = df['coordY'] / (df['coordZ'] + 1e-6)

# size
df['diameter_category'] = pd.cut(df['diameter_mm'], 
                                  bins=[0, 5, 10, 15, 20, 100], 
                                  labels=['very small', 'small', 'medium', 'large', 'very large'])

# Log of size
df['log_diameter'] = np.log1p(df['diameter_mm'])

print(df[['x_abs', 'y_abs', 'z_abs', 'distance_from_origin', 'log_diameter']].head())

# %% [markdown]
# # 3. Generate negative data (non-nodules)

# %%

n_negative = len(df) * 3

np.random.seed(42)

# Get the distribution of positive data.
mean_x, std_x = df['coordX'].mean(), df['coordX'].std()
mean_y, std_y = df['coordY'].mean(), df['coordY'].std()
mean_z, std_z = df['coordZ'].mean(), df['coordZ'].std()
mean_d, std_d = df['diameter_mm'].mean(), df['diameter_mm'].std()

# Generate negative data (slightly more noise) 
negative_data = {
    'coordX': np.random.normal(mean_x * 1.5, std_x * 2, n_negative),
    'coordY': np.random.normal(mean_y * 1.5, std_y * 2, n_negative),
    'coordZ': np.random.normal(mean_z * 1.5, std_z * 2, n_negative),
    'diameter_mm': np.random.exponential(mean_d, n_negative),  
    'is_nodule': 0  # Label 0: Not a nodule 
}

df_negative = pd.DataFrame(negative_data)

# Create similar features for negative data.
df_negative['x_abs'] = np.abs(df_negative['coordX'])
df_negative['y_abs'] = np.abs(df_negative['coordY'])
df_negative['z_abs'] = np.abs(df_negative['coordZ'])
df_negative['distance_from_origin'] = np.sqrt(df_negative['coordX']**2 + 
                                               df_negative['coordY']**2 + 
                                               df_negative['coordZ']**2)
df_negative['xy_ratio'] = df_negative['coordX'] / (df_negative['coordY'] + 1e-6)
df_negative['xz_ratio'] = df_negative['coordX'] / (df_negative['coordZ'] + 1e-6)
df_negative['yz_ratio'] = df_negative['coordY'] / (df_negative['coordZ'] + 1e-6)
df_negative['log_diameter'] = np.log1p(df_negative['diameter_mm'])

print(f"Negative sample created for {n_negative}")

# %% [markdown]
# # 4. Combining positive and negative data

# %%
df_positive = df.copy()
df_positive['is_nodule'] = 1

# Select featured columns
feature_columns = ['x_abs', 'y_abs', 'z_abs', 'distance_from_origin', 
                   'xy_ratio', 'xz_ratio', 'yz_ratio', 'log_diameter']

df_combined = pd.concat([df_positive[feature_columns + ['is_nodule']], 
                         df_negative[feature_columns + ['is_nodule']]], 
                        ignore_index=True)

print(f"Total sample: {len(df_combined)}")
print(f"Number of positive samples (with nodules): {len(df_positive)}")
print(f"Number of negative samples (with nodules): {len(df_negative)}")
print(f"\nImbalance rate: 1:{len(df_negative)//len(df_positive):.0f}")

# Separate features and labels
X = df_combined[feature_columns]
y = df_combined['is_nodule']

# Standardize 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_columns)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, 
                                                      random_state=42, stratify=y)

print(f"\nTrain: {X_train.shape}")
print(f"Test: {X_test.shape}")

# %% [markdown]
# # 5. Model Random Forest 

# %%
rf_model = RandomForestClassifier(
    n_estimators=100,        
    max_depth=10,            
    min_samples_split=5,    
    min_samples_leaf=2,      
    class_weight='balanced', 
    random_state=42,
    n_jobs=-1                
)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]


# %% [markdown]
# # 6. Model Evaluation 

# %%
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# AUC-ROC
auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC-ROC: {auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print("                 Predicted")
print("                 Negative    Positive")
print(f"Actual Negative  {cm[0,0]:5d}  {cm[0,1]:5d}")
print(f"Actual Positive  {cm[1,0]:5d}  {cm[1,1]:5d}")

# Calculate detailed metrics
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)  # True positive detection rate
specificity = tn / (tn + fp)  # True negative detection rate
precision = tp / (tp + fp)
f1 = 2 * (precision * sensitivity) / (precision + sensitivity)

print(f"\nDetailed metrics:")
print(f"- Sensitivity (Recall): {sensitivity:.4f} (correctly detects {sensitivity*100:.1f}% of positives)")
print(f"- Specificity: {specificity:.4f} (correctly identifies {specificity*100:.1f}% of negatives)")
print(f"- Precision: {precision:.4f}")
print(f"- F1-score: {f1:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Nodule', 'Nodule']))

# %% [markdown]
# # 7. Visualization

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 7.1. Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0],
            xticklabels=['No Nodule', 'Nodule'],
            yticklabels=['No Nodule', 'Nodule'])
axes[0,0].set_title('Confusion Matrix')
axes[0,0].set_ylabel('Actual')
axes[0,0].set_xlabel('Predicted')

# 7.2. Feature Importance
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = [feature_columns[i] for i in indices]

axes[0,1].barh(range(len(indices)), importances[indices])
axes[0,1].set_yticks(range(len(indices)))
axes[0,1].set_yticklabels(feature_names)
axes[0,1].set_xlabel('Importance')
axes[0,1].set_title('Most Important Features')

# 7.3. Predicted Probability Distribution
axes[1,0].hist(y_pred_proba[y_test==0], bins=30, alpha=0.7, label='No Nodule', color='blue')
axes[1,0].hist(y_pred_proba[y_test==1], bins=30, alpha=0.7, label='Nodule', color='red')
axes[1,0].set_xlabel('Predicted Probability')
axes[1,0].set_ylabel('Count')
axes[1,0].set_title('Predicted Probability Distribution')
axes[1,0].legend()

# 7.4. Nodule Size Comparison
axes[1,1].hist(df_positive['diameter_mm'], bins=30, alpha=0.7, label='True Nodules', color='red')
axes[1,1].hist(df_negative['diameter_mm'], bins=30, alpha=0.7, label='Background', color='blue')
axes[1,1].set_xlabel('Diameter (mm)')
axes[1,1].set_ylabel('Count')
axes[1,1].set_title('Size Distribution')
axes[1,1].legend()

plt.tight_layout()

# Save figure to file
plt.savefig("model_evaluation_plots.png", dpi=300, bbox_inches='tight')

plt.show()

# %% [markdown]
# # 8. Predict on Some Random Samples

# %%
# Select 10 random samples from the test set
n_samples = 10
sample_indices = np.random.choice(len(X_test), n_samples, replace=False)

for i, idx in enumerate(sample_indices):
    true_label = y_test.iloc[idx]
    pred_label = y_pred[idx]
    pred_proba = y_pred_proba[idx]
    
    status = "CORRECT" if true_label == pred_label else "INCORRECT"
    
    print(f"Sample {i+1}:")
    print(f"  - Actual: {'NODULE' if true_label == 1 else 'NO NODULE'}")
    print(f"  - Predicted: {'NODULE' if pred_label == 1 else 'NO NODULE'} (probability: {pred_proba:.3f})")
    print(f"  - Result: {status}")
    print()


