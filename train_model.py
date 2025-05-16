import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight  # Import class weight function

# Upload dataset
df = pd.read_csv('complaint-dataset.csv', encoding='ISO-8859-1')

# Drop rows where 'Category' or 'Grievance Description' is missing
df = df.dropna(subset=['Category', 'Grievance Description'])

# Fill missing descriptions with a placeholder (if any remain)
df['Grievance Description'] = df['Grievance Description'].fillna("Unknown Complaint")

# Define features and target
X = df['Grievance Description'].astype(str)  # Ensure it's a string
y = df['Category']

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Train Random Forest with class weights
clf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    max_depth=20,
    min_samples_split=5,
    class_weight=class_weight_dict
)
clf.fit(X_train_tfidf, y_train)

# Predict on test set
y_pred = clf.predict(X_test_tfidf)

# Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Get unique class labels in y_test
unique_classes_in_test = np.unique(y_test)

# Map the numerical labels back to category names
target_names = [label_encoder.classes_[i] for i in unique_classes_in_test]

# Generate classification report with correct labels
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Confusion Matrix
'''plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()'''


cm = confusion_matrix(y_test, y_pred)
unique_classes_in_test = np.unique(y_test)

# Map the numerical labels back to category names for the test set
target_names_test = [label_encoder.classes_[i] for i in unique_classes_in_test]

# Create the DataFrame with the correct labels for the test set
cm_df = pd.DataFrame(cm, index=target_names_test, columns=target_names_test)

print("\nConfusion Matrix (Labeled):")
print(cm_df)
