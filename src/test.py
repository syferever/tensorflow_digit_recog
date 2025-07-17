from keras import models
from .load_data import load_images
import matplotlib.pyplot as plt


PATH = "data_gas_pic/"

loaded_model = models.load_model("img_rec.keras")
loaded_model.summary()

images, labels = load_images(PATH)
images = images.reshape(-1, 50, 90, 1).astype("float32") / 255.0

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images[i], cmap="gray")
    plt.xlabel(labels[i])
plt.show()

eval_results = loaded_model.evaluate(
    images,
    {
        "prediction_digit1": labels[:, 0],
        "prediction_digit2": labels[:, 1],
        "prediction_digit3": labels[:, 2],
    },
    verbose=2,
)

print(f"\nTotal Test Loss: {eval_results[0]:.4f}")
print(f"Digit 1 Test Accuracy: {eval_results[4]:.4f}")  # Adjust index based on summary
print(f"Digit 2 Test Accuracy: {eval_results[5]:.4f}")
print(f"Digit 3 Test Accuracy: {eval_results[6]:.4f}")
