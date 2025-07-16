import matplotlib.pyplot as plt

# import numpy as np
import tensorflow as tf
import keras
from keras import layers, models

(train_images, train_labels), (test_images, test_labels) = (
    tf.keras.datasets.mnist.load_data()
)

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
print(f"\nShape of training images after reshaping: {train_images.shape}")
print(f"Shape of test images after reshaping: {test_images.shape}")

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap="gray")
    plt.xlabel(train_labels[i])
plt.show()

model = models.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPool2D(),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPool2D(),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.Dropout(0.8),
        layers.Flatten(),
        layers.Dense(10),
    ]
)

print(model.summary())

model.compile(
    optimizer=keras.optimizers.Adagrad(0.05),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

history = model.fit(
    train_images,
    train_labels,
    batch_size=64,
    epochs=10,
    validation_data=(test_images, test_labels),
)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim([0, 1])
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.show()

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nTest accuracy: {test_acc}")
