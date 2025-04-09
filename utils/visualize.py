import matplotlib.pyplot as plt
import numpy as np
import torch

def show_images(original_images, adversarial_images, filename="adversarial_results.png"):
    original_images = original_images.detach().cpu().numpy()
    adversarial_images = adversarial_images.detach().cpu().numpy()

    num_images = min(len(original_images), 5)
    fig, axes = plt.subplots(num_images, 2, figsize=(6, num_images * 3))

    for i in range(num_images):
        orig = np.transpose(original_images[i], (1, 2, 0))
        adv = np.transpose(adversarial_images[i], (1, 2, 0))

        axes[i, 0].imshow(orig)
        axes[i, 0].axis('off')
        axes[i, 0].set_title("Original")

        axes[i, 1].imshow(adv)
        axes[i, 1].axis('off')
        axes[i, 1].set_title("Adversarial")

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Images saved as {filename}")

# Example usage:
# save_images(original_images, adversarial_images)
