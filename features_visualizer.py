import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import min_max_scaler

video_names = ["test_5", "test_6", "Video_1"]
gt = {"test_5": [2800, 7870], "test_6": [2776, 8320], "Video_1": [939, 5438]}

for video_name in video_names:
    file = f"{video_name}_all.npy"
    if not os.path.isfile(file):
        break
    with open(file, "rb") as f:
        features = np.load(f, allow_pickle=True)
    print(features.shape)

    features = min_max_scaler(features)
    total_imgs = len(features)
    diff_features = ((features[: total_imgs - 1, :] - features[1:, :]) ** 2) ** 0.5
    total_diff_features = len(diff_features)

    # diff
    for i in range(len(diff_features[0])):
        x = np.arange(total_diff_features)
        plt.plot(x, diff_features[:, i])
        plt.scatter(gt[video_name][0], 0, c="#1f33b4")
        plt.scatter(gt[video_name][1], 0, c="#1f33b4")
        plt.savefig(f"features/{video_name}/all/{i}_diff.png")
        plt.clf()

    # origin
    for i in range(len(features[0])):
        x = np.arange(total_imgs)
        plt.plot(x, features[:, i])
        plt.scatter(gt[video_name][0], 0, c="#1f33b4")
        plt.scatter(gt[video_name][1], 0, c="#1f33b4")
        plt.savefig(f"features/{video_name}/all/{i}.png")
        plt.clf()
