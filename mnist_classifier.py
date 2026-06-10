import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# Normalize pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build a simple neural network
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evaluate on test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")

# --- Visualization of predictions ---
# Pick 5 random test images
import numpy as np
indices = np.random.choice(len(x_test), 5)

for i in indices:
    img = x_test[i]
    label = y_test[i]
    prediction = model.predict(img.reshape(1, 28, 28))
    predicted_label = prediction.argmax()

    plt.imshow(img, cmap='gray')
    plt.title(f"True: {label}, Predicted: {predicted_label}")
    plt.show()
