import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Function to load data and extract features
def load_data(data_dir):
    X, y = [], []
    for filename in os.listdir(data_dir):
        if filename.endswith('.jpg'):
            image_path = os.path.join(data_dir, filename)
            X.append(extract_features(image_path))
            # Extract the label from the filename
            label = filename.split('_')[0]  # Assuming file names are labeled as 'male_123.jpg' or 'female_456.jpg'
            if label == 'male':
                y.append(0)  # 0 for male
            else:
                y.append(1)  # 1 for female
    return np.array(X), np.array(y)

# Function to extract features from images
def extract_features(image_path):
    # Read the image
    img = cv2.imread(image_path)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resize the image to a fixed size
    resized_img = cv2.resize(gray, (150, 150))
    # Flatten the image
    flattened_img = resized_img.flatten()
    return flattened_img

# Load data
data_dir = 'Data/Bollywood-Data-master/images-data/dir_001'  # Path to directory containing movie poster images
X, y = load_data(data_dir)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the features for CNN input (assuming grayscale images)
X_train = X_train.reshape(X_train.shape[0], 150, 150, 1)
X_test = X_test.reshape(X_test.shape[0], 150, 150, 1)

# Define the model architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification (0: male, 1: female)
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)

# Predict on test set
y_pred = model.predict(X_test)

# Count occurrences of male and female in predictions
male_count = np.sum(y_pred < 0.5)  # Count instances where prediction is less than 0.5
female_count = len(y_pred) - male_count  # Count instances where prediction is 0.5 or higher

# Plot bar graph
labels = ['Male', 'Female']
counts = [male_count, female_count]

plt.bar(labels, counts, color=['blue', 'pink'])
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Appearances of Male and Female in Movie Posters')
plt.show()
