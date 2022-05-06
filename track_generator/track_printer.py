import cv2

from map_generator import generate_random_track
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

fig, axs = plt.subplots(2, 3, figsize=(10, 5))

for i in range(2):
    for j in tqdm(range(3)):
        image = generate_random_track()
        axs[i, j].imshow(image, cmap='gray')
        axs[i, j].axis('off')
plt.show()