import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Load your dataset 
data = pd.read_csv('credit_data.csv') # chage where your csv file is

# Encode categorical variables
categorical_cols = ['credit_history', 'employment_status']
le = LabelEncoder()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

target_col = 'creditworthy'
X = data.drop(target_col, axis=1)
y = le.fit_transform(data[target_col])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print(f"\n{name} Results")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# Hyperparameter tuning for Gradient Boosting
param_grid = {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}
grid_search = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=3)
grid_search.fit(X_train_scaled, y_train)
print("\nBest Parameters (Gradient Boosting):", grid_search.best_params_)
