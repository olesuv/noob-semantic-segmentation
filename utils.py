import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os


def load_images(image_dir, mask_dir):
    images = []
    masks = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.coco.json'):
            continue
        if filename == '2369_jpg.rf.8b8afa9d79c61fa42ca128c940b9cbc0.jpg':
            os.remove(f"{image_dir}/filename")

        img = tf.keras.preprocessing.image.load_img(
            os.path.join(image_dir, filename), target_size=(128, 128))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img /= 255.0
        images.append(img)

        mask = tf.keras.preprocessing.image.load_img(os.path.join(mask_dir, filename),
                                                     color_mode="grayscale", target_size=(128, 128))
        mask = tf.keras.preprocessing.image.img_to_array(mask)
        mask /= 255.0
        masks.append(mask)

    return np.array(images), np.array(masks)


def dice_score(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) +
                                           smooth)


def save_results_img(input_img, predicted_img, true_img, n: int):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(input_img.squeeze(), cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    axes[1].imshow(predicted_img.squeeze(), cmap='gray')
    axes[1].set_title('Predicted Image')
    axes[1].axis('off')

    axes[2].imshow(true_img.squeeze(), cmap='gray')
    axes[2].set_title('True Image')
    axes[2].axis('off')

    plt.savefig(f"./benchmarks/results_{n}.png")
    plt.show()
