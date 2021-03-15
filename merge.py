import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

features_index = 20
video_names = ["test_5", "Video_1", "test_6"]

for video_name in video_names:
    file1 = f"{video_name}_features.npy"
    file2 = f"{video_name}_features_{features_index}.npy"

    with open(file1, "rb") as f:
        features1 = np.load(f, allow_pickle=True)

    with open(file2, "rb") as f:
        features2 = np.load(f, allow_pickle=True)

    features1 = np.array(features1, dtype=object)
    features2 = np.array(features2, dtype=object)

    features = np.concatenate((features1, features2), axis=1)
    print(features.shape)

    with open(f"{video_name}_all.npy", "wb") as f:
        np.save(f, features)
