# from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras import layers
from .load_data import load_images

PATH = "digits/"


def create_multi_digit_image(images, labels, num_digits=3, num_samples=5000):
    multi_images = []
    multi_labels = []

    for _ in range(num_samples):
        indices = np.random.choice(len(images), num_digits, replace=False)
        selected_images = [images[i] for i in indices]
        selected_labels = [labels[i] for i in indices]

        concatenated_image = np.concatenate(selected_images, axis=1)

        multi_images.append(concatenated_image)
        multi_labels.append(selected_labels)

    return np.array(multi_images), np.array(multi_labels)


images, labels = load_images(PATH)
train_images, train_labels = create_multi_digit_image(
    images, labels, num_digits=3, num_samples=20000
)
test_images, test_labels = create_multi_digit_image(
    images, labels, num_digits=3, num_samples=20000
)

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap="gray")
    plt.xlabel(train_labels[i])
plt.show()

# (mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()
# mnist_images = mnist_images.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# print("Generating synthetic multi-digit dataset...")
# train_images, train_labels = create_multi_digit_image(mnist_images, mnist_labels, num_digits=3, num_samples=20000)
# test_images, test_labels = create_multi_digit_image(num_digits=3)

# images, labels = load_images(PATH)
# images = images.reshape(-1, 50, 90, 1).astype("float32") / 255.0

# train_images, test_images, train_labels, test_labels = train_test_split(
#     images,
#     labels,
#     test_size=0.2,
#     random_state=42,
#     stratify=labels,
# )

print(f"\nShape of training images after reshaping: {train_images.shape}")
print(f"Shape of test images after reshaping: {test_images.shape}")


input_layer = layers.Input(shape=(50, 90, 1), name="input_image")
cnn_sequential = layers.Conv2D(32, (3, 3), activation="relu")(input_layer)
cnn_sequential = layers.MaxPool2D()(cnn_sequential)
cnn_sequential = layers.Conv2D(64, (3, 3), activation="relu")(cnn_sequential)
cnn_sequential = layers.MaxPool2D()(cnn_sequential)
cnn_sequential = layers.Conv2D(64, (3, 3), activation="relu")(cnn_sequential)
cnn_sequential = layers.Dropout(0.8)(cnn_sequential)
cnn_sequential = layers.Flatten()(cnn_sequential)

softmax1 = layers.Dense(128)(cnn_sequential)
softmax1 = layers.Dense(11)(softmax1)
softmax1 = layers.Softmax(name="prediction_digit1")(softmax1)
softmax2 = layers.Dense(128)(cnn_sequential)
softmax2 = layers.Dense(11)(softmax2)
softmax2 = layers.Softmax(name="prediction_digit2")(softmax2)
softmax3 = layers.Dense(128)(cnn_sequential)
softmax3 = layers.Dense(11)(softmax3)
softmax3 = layers.Softmax(name="prediction_digit3")(softmax3)

model = keras.Model(inputs=input_layer, outputs=[softmax1, softmax2, softmax3])

print(model.summary())

model.compile(
    optimizer=keras.optimizers.Adagrad(0.05),
    loss={
        "prediction_digit1": "sparse_categorical_crossentropy",
        "prediction_digit2": "sparse_categorical_crossentropy",
        "prediction_digit3": "sparse_categorical_crossentropy",
    },
    metrics={
        "prediction_digit1": "accuracy",
        "prediction_digit2": "accuracy",
        "prediction_digit3": "accuracy",
    },
)

print("\n--- Training the multi-digit model ---")
history = model.fit(
    train_images,
    {
        "prediction_digit1": train_labels[:, 0],
        "prediction_digit2": train_labels[:, 1],
        "prediction_digit3": train_labels[:, 2],
    },
    batch_size=64,
    epochs=10,
    validation_data=(
        test_images,
        {
            "prediction_digit1": test_labels[:, 0],
            "prediction_digit2": test_labels[:, 1],
            "prediction_digit3": test_labels[:, 2],
        },
    ),
)


print("\n--- Evaluating the model ---")
eval_results = model.evaluate(
    test_images,
    {
        "prediction_digit1": test_labels[:, 0],
        "prediction_digit2": test_labels[:, 1],
        "prediction_digit3": test_labels[:, 2],
    },
    verbose=2,
)

print(f"\nTotal Test Loss: {eval_results[0]:.4f}")
print(f"Digit 1 Test Accuracy: {eval_results[4]:.4f}")  # Adjust index based on summary
print(f"Digit 2 Test Accuracy: {eval_results[5]:.4f}")
print(f"Digit 3 Test Accuracy: {eval_results[6]:.4f}")

model.save("img_rec.keras")
print(f"Model saved successfully to: {'img_rec.keras'}")
