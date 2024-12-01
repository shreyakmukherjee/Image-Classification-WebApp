import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to the range [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Convert labels to categorical (one-hot encoding)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Resize images to match MobileNetV2's input size (224x224)
X_train_resized = tf.image.resize(X_train, (224, 224))
X_test_resized = tf.image.resize(X_test, (224, 224))

# Load the MobileNetV2 model pre-trained on ImageNet
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,  # Exclude the top layer
    weights="imagenet"
)

# Freeze the base model to retain pre-trained weights
base_model.trainable = False

# Add custom classification layers
model = models.Sequential([
    base_model,  # Pre-trained MobileNetV2
    layers.GlobalAveragePooling2D(),  # Global average pooling
    layers.Dense(128, activation="relu"),  # Fully connected layer
    layers.Dropout(0.5),  # Dropout for regularization
    layers.Dense(10, activation="softmax")  # Output layer for 10 classes
])

# Compile the model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Display the model summary
model.summary()

# Train the model
history = model.fit(
    X_train_resized, y_train,
    validation_data=(X_test_resized, y_test),
    epochs=100,
    batch_size=32
)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test_resized, y_test)
print(f"Test Accuracy: {test_acc:.2f}")

# Plot training and validation accuracy
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")
plt.show()

# Make predictions
predictions = model.predict(X_test_resized)

# Map class labels to their names
class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

# Visualize predictions
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_test[i])
    pred_label = class_names[predictions[i].argmax()]
    true_label = class_names[y_test[i].argmax()]
    plt.title(f"Pred: {pred_label}\nTrue: {true_label}")
    plt.axis("off")
plt.show()