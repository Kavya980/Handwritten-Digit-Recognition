from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import joblib


print("Loading MNIST dataset...")

# Load MNIST dataset
mnist = fetch_openml(
    "mnist_784",
    version=1,
    as_frame=False
)

X = mnist.data
y = mnist.target.astype(np.int8)

print("Dataset loaded!")
print("Total images:", len(X))


# Convert pixel values from 0-255 to 0-1
X = X / 255.0


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    random_state=42,
    stratify=y
)


print("Training images:", len(X_train))
print("Testing images:", len(X_test))


# Create SVM model
print("Training SVM model...")

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)


# Train model
model.fit(X_train, y_train)


print("Model trained!")


# Test model
print("Testing model...")

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("Model Accuracy:", accuracy)


# Save model
joblib.dump(
    model,
    "digit_model.pkl"
)

print("Model saved successfully!")