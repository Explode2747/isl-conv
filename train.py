# train_alphabet_model.py

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

# Paths
dataset_dir = 'ISL_Dataset'  # Path to your dataset directory

# Data augmentation and normalization
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Training set
train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(64, 64),  # Resize all images to 64x64
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Validation set
validation_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    epochs=10,  # Adjust the number of epochs based on your dataset
    validation_data=validation_generator
)

# Save the trained model
model.save('alphabet_classification_model.h5')

# Save the class labels
labels = list(train_generator.class_indices.keys())
with open('labels.json', 'w') as f:
    json.dump(labels, f)

print("Model training complete and saved as 'alphabet_classification_model.h5' with labels.")
