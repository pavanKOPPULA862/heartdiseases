# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc

# Step 2: Load the dataset
data = pd.read_csv('heart_disease.csv')  # Replace with your dataset path
print(data.head())  # Display the first few rows

# Step 3: Data Preprocessing
X = data.drop(columns=['target'])  # Features
y = data['target']  # Target column

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Train Models - Logistic Regression and Decision Tree Classifier
# Logistic Regression Model
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)
log_preds = log_model.predict(X_test_scaled)

# Decision Tree Model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)

# Step 5: Evaluate Models

# Logistic Regression Evaluation
log_accuracy = accuracy_score(y_test, log_preds)
log_conf_matrix = confusion_matrix(y_test, log_preds)
log_class_report = classification_report(y_test, log_preds)

# Decision Tree Evaluation
dt_accuracy = accuracy_score(y_test, dt_preds)
dt_conf_matrix = confusion_matrix(y_test, dt_preds)
dt_class_report = classification_report(y_test, dt_preds)

# Step 6: ROC Curves
log_fpr, log_tpr, _ = roc_curve(y_test, log_model.predict_proba(X_test_scaled)[:, 1])
log_auc = auc(log_fpr, log_tpr)

dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_model.predict_proba(X_test)[:, 1])
dt_auc = auc(dt_fpr, dt_tpr)

# Step 7: Plotting the Dashboard using Matplotlib

fig, axs = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Accuracy Comparison
axs[0, 0].bar(['Logistic Regression', 'Decision Tree'], [log_accuracy, dt_accuracy], color=['blue', 'green'])
axs[0, 0].set_title('Model Accuracy Comparison')
axs[0, 0].set_ylabel('Accuracy')
axs[0, 0].set_ylim([0, 1])

# Plot 2: Confusion Matrix - Logistic Regression
sns.heatmap(log_conf_matrix, annot=True, fmt='d', cmap='Blues', ax=axs[0, 1])
axs[0, 1].set_title('Logistic Regression Confusion Matrix')
axs[0, 1].set_xlabel('Predicted')
axs[0, 1].set_ylabel('Actual')

# Plot 3: Confusion Matrix - Decision Tree
sns.heatmap(dt_conf_matrix, annot=True, fmt='d', cmap='Blues', ax=axs[1, 0])
axs[1, 0].set_title('Decision Tree Confusion Matrix')
axs[1, 0].set_xlabel('Predicted')
axs[1, 0].set_ylabel('Actual')

# Plot 4: ROC Curves
axs[1, 1].plot(log_fpr, log_tpr, color='darkorange', lw=2, label=f'Logistic Regression AUC = {log_auc:.2f}')
axs[1, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axs[1, 1].plot(dt_fpr, dt_tpr, color='blue', lw=2, label=f'Decision Tree AUC = {dt_auc:.2f}')
axs[1, 1].set_xlim([0.0, 1.0])
axs[1, 1].set_ylim([0.0, 1.05])
axs[1, 1].set_xlabel('False Positive Rate')
axs[1, 1].set_ylabel('True Positive Rate')
axs[1, 1].set_title('ROC Curve Comparison')
axs[1, 1].legend(loc="lower right")

plt.tight_layout()
plt.show()

