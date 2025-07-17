import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import glob


def load_images(PATH, show_examples=False):
    images = []
    labels = []

    for filepath in glob.glob(PATH + "*.webp"):
        image = Image.open(filepath)
        gray_image = image.convert("L")
        images.append(np.array(gray_image))
        # digits = int(Path(filepath).name.rsplit(".", 1)[0])
        digits = list(map(int, str(int(Path(filepath).name.rsplit("_", 1)[0]))))
        if len(digits) < 2:
            digits.insert(0, 10)
        digits.extend([0])
        labels.append(digits)

    if show_examples:
        fig, axes = plt.subplots(2, 5, figsize=(10, 4))
        for idx, ax in enumerate(axes.flat):
            if idx < len(images):
                ax.imshow(images[idx], cmap="gray")
                ax.set_title(f"Label: {labels[idx]}")
                ax.axis("off")
        plt.tight_layout()
        plt.show()
    return np.array(images), np.array(labels)
