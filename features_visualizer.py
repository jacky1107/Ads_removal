import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import min_max_scaler

video_names = ["TVshow_7"]  # , "test_6", "Video_1"]
gt = {"TVshow_7": [[14389, 18143], [28133, 31893]]}
for video_name in video_names:
    file = f"{video_name}_features.npy"
    if not os.path.isfile(file):
        break
    with open(file, "rb") as f:
        features = np.load(f, allow_pickle=True)
    print(features.shape)

    features = min_max_scaler(features)
    total_imgs = len(features)
    diff_features = (features[: total_imgs - 1, :] -
                     features[1:, :]) ** 2  # ** 0.5
    diff_features = min_max_scaler(diff_features)
    total_diff_features = len(diff_features)

    # diff
    for i in range(len(diff_features[0])):
        x = np.arange(total_diff_features)
        plt.plot(x, diff_features[:, i])
        mean = np.mean(diff_features[:, i])
        for segs in gt[video_name]:
            for x in segs:
                plt.axvline(x=x, color="r", linestyle=":")

        plt.savefig(f"features/{video_name}/all/{i}_diff.png")
        plt.clf()

    # origin
    for i in range(len(features[0])):
        x = np.arange(total_imgs)
        plt.plot(x, features[:, i])
        mean = np.mean(features[:, i])
        for segs in gt[video_name]:
            for x in segs:
                plt.axvline(x=x, color="r", linestyle=":")
        plt.savefig(f"features/{video_name}/all/{i}.png")
        plt.clf()
