# FILE 2: main.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data
print("Loading data...")
df = pd.read_csv('data/wine_quality.csv')
print(f"Dataset shape: {df.shape}\n")

# Separate features and target
X = df.drop('quality', axis=1)
y = df['quality']

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVC': SVC(kernel='rbf'),
    'SGD': SGDClassifier(loss='log_loss', random_state=42)
}

print("="*50)
print("MODEL RESULTS")
print("="*50)

results = {}
for name, model in models.items():
    print(f"\n{name}:")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Quality distribution
axes[0, 0].hist(df['quality'], bins=10, color='steelblue', edgecolor='black')
axes[0, 0].set_title('Quality Distribution')
axes[0, 0].set_xlabel('Quality')
axes[0, 0].set_ylabel('Frequency')

# 2. Correlation heatmap
sns.heatmap(df.corr(), ax=axes[0, 1], cmap='coolwarm', cbar=False)
axes[0, 1].set_title('Feature Correlation')

# 3. Model comparison
axes[1, 0].bar(results.keys(), results.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
axes[1, 0].set_title('Model Accuracy Comparison')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].set_ylim([0, 1])

# 4. Feature importance (Random Forest)
rf_model = models['Random Forest']
importances = rf_model.feature_importances_
top_features = np.argsort(importances)[-5:]
axes[1, 1].barh(X.columns[top_features], importances[top_features], color='coral')
axes[1, 1].set_title('Top 5 Feature Importance')

plt.tight_layout()
plt.savefig('results.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as 'results.png'")
plt.show()

print("\n" + "="*50)
print(f"Best Model: {max(results, key=results.get)}")
print("="*50)