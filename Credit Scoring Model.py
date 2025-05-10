import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df = pd.read_csv("train.csv", low_memory=False)

# Display the first few rows of the dataframe
print("First few rows of the dataset:")
print(df.head())

# Checking the size of the dataset
print(f"\nDataset Size: {df.shape}")
print(f"Number of rows: {df.shape[0]} | Number of columns: {df.shape[1]}")

# Columns present in the dataset
print("\nColumns in the dataset:")
print(df.columns)

# Data types of each column
print("\nData types of columns:")
print(df.dtypes)

# Checking for missing values
print("\nMissing Values in each column:")
print(df.isnull().sum())

# Dropping unnecessary columns
df.drop(["ID", "Customer_ID", "Name", "SSN", "Type_of_Loan"], axis=1, inplace=True)

# Check if columns were dropped successfully
print("\nColumns after dropping unnecessary ones:")
print(df.columns)

# Identifying numerical and categorical columns
cat_cols = [feature for feature in df.columns if df[feature].dtype == 'O']
num_cols = [feature for feature in df.columns if feature not in cat_cols]

print(f"\nCategorical Columns: {cat_cols}")
print(f"Numerical Columns: {num_cols}")

# Checking unique values in categorical columns
print("\nUnique values in categorical columns:")
for feature in cat_cols:
    print(f"\n{feature}:")
    print(f"Number of unique values: {df[feature].nunique()}")
    print(f"Unique values: {df[feature].unique()}")

# Handling missing values in categorical columns
print("\nHandling missing values in categorical columns:")
for feature in cat_cols:
    missing_count = df[feature].isnull().sum()
    print(f"{feature} - Missing values: {missing_count}")

# Handling missing values in numerical columns
print("\nHandling missing values in numerical columns:")
for feature in num_cols:
    missing_count = df[feature].isnull().sum()
    print(f"{feature} - Missing values: {missing_count}")

# Imputing missing values for numerical columns with median
imputer = SimpleImputer(strategy='median')
df[num_cols] = imputer.fit_transform(df[num_cols])

# Imputing missing values for categorical columns with the most frequent value
imputer_cat = SimpleImputer(strategy='most_frequent')
df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

# Verifying that missing values have been imputed
print("\nMissing values after imputation:")
print(df.isnull().sum())

# Encoding categorical columns using Label Encoding
print("\nEncoding categorical columns:")
label_encoder = LabelEncoder()
for feature in cat_cols:
    df[feature] = label_encoder.fit_transform(df[feature])

# Verifying encoding of categorical columns
print("\nEncoded categorical columns:")
print(df[cat_cols].head())

# Splitting the dataset into training and testing sets
X = df.drop("Credit_Score", axis=1)
y = df["Credit_Score"]

# Encoding the target variable
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"\nTraining Data Size: {X_train.shape[0]} | Testing Data Size: {X_test.shape[0]}")

# Feature Scaling (if necessary, you can scale numerical features)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Building a Random Forest Classifier model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Model Evaluation
print("\nModel Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Feature Importance
feature_importances = rf_model.feature_importances_
sorted_idx = np.argsort(feature_importances)[::-1]

# Plotting feature importance
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align="center")
plt.yticks(range(len(sorted_idx)), [X.columns[i] for i in sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Feature Importance in Random Forest Model')
plt.show()

# Saving the trained model for future use
import joblib
joblib.dump(rf_model, 'credit_score_predictor_model.pkl')

# Loading the model back and making predictions
loaded_model = joblib.load('credit_score_predictor_model.pkl')
new_predictions = loaded_model.predict(X_test)

print("\nPredictions from the loaded model:")
print(new_predictions[:10])

# Visualizing the distribution of numerical features
print("\nVisualizing the distribution of numerical features:")
df[num_cols].hist(figsize=(12, 10), bins=20)
plt.suptitle('Distribution of Numerical Features')
plt.show()

# Visualizing the correlation between numerical features
correlation_matrix = df[num_cols].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Exploring the distribution of the target variable
print("\nDistribution of the target variable (Credit Score):")
print(df["Credit_Score"].value_counts())

# Visualizing the distribution of the target variable
plt.figure(figsize=(8, 6))
sns.countplot(x="Credit_Score", data=df, palette="viridis")
plt.title('Distribution of Credit Score')
plt.show()

# Handling outliers in numerical features (optional, based on business understanding)
# Here, we can use Z-score or IQR to detect and remove outliers

from scipy import stats

# Detecting and removing outliers using Z-score
z_scores = np.abs(stats.zscore(df[num_cols]))
df_no_outliers = df[(z_scores < 3).all(axis=1)]

print(f"\nData after removing outliers: {df_no_outliers.shape}")

# Rebuilding the model without outliers
X_no_outliers = df_no_outliers.drop("Credit_Score", axis=1)
y_no_outliers = df_no_outliers["Credit_Score"]
X_train_no_outliers, X_test_no_outliers, y_train_no_outliers, y_test_no_outliers = train_test_split(X_no_outliers, y_no_outliers, test_size=0.3, random_state=42)

# Training the model
rf_model_no_outliers = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_no_outliers.fit(X_train_no_outliers, y_train_no_outliers)

# Evaluating the model
y_pred_no_outliers = rf_model_no_outliers.predict(X_test_no_outliers)
print(f"\nAccuracy after removing outliers: {accuracy_score(y_test_no_outliers, y_pred_no_outliers):.4f}")

