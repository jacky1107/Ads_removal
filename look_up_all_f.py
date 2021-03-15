import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from utils import *


save_feature = False
video_names = ["test_5", "test_6", "Video_1"]

txt = ""
for video_name in video_names:
    gt = {"test_5": [2800, 7870], "test_6": [2776, 8320], "Video_1": [939, 5438]}
    with open(f"{video_name}_features.npy", "rb") as f:
        features = np.load(f, allow_pickle=True)
    print(features.shape)

    total_imgs = len(features)
    diff_features = ((features[: total_imgs - 1, :] - features[1:, :]) ** 2) ** 0.5
    total_diff_features = len(diff_features)

    features_index_thres = {
        "test_5": [
            (diff_features, 18, 750, 30),
            (diff_features, 19, 500, 30),
        ],
        "test_6": [
            (diff_features, 18, 650, 30),
            (diff_features, 19, 280, 30),
        ],
        "Video_1": [
            # (diff_features, 10, 0.3, 30),
            (diff_features, 13, 330, 30),
            (diff_features, 18, 600, 30),
            # (diff_features, 19, 350, 30),
        ],
    }

    if save_feature:
        # diff
        total_diff_features = len(diff_features)
        for i in range(len(diff_features[0])):
            upper = total_diff_features - 1
            x = np.arange(0, upper, upper / total_diff_features)
            plt.plot(x, diff_features[:, i])
            plt.scatter(gt[video_name][0], 0, c="#1f33b4")
            plt.scatter(gt[video_name][1], 0, c="#1f33b4")
            plt.savefig(f"features_local/{video_name}_{i}_diff.png")
            plt.clf()

        # origin
        for i in range(len(features[0])):
            x = np.arange(total_imgs)
            plt.plot(x, features[:, i])
            plt.scatter(gt[video_name][0], 0, c="#1f33b4")
            plt.scatter(gt[video_name][1], 0, c="#1f33b4")
            plt.savefig(f"features_local/{video_name}_{i}.png")
            plt.clf()

    all_res = []
    for f_index_thres in features_index_thres[video_name]:
        index = f_index_thres[1]
        new = get_segmentation(
            f_index_thres[0],
            f_index_thres[1],
            f_index_thres[2],
        )
        print(new)
        assert len(new) % 2 == 0
        new = new.reshape((len(new) // 2, 2))
        new = merge_small_seg(new, f_index_thres[3])
        new = merge_gap_between_seg(new, f_index_thres[3])
        print(new)

        upper = total_diff_features
        x = np.arange(upper)
        y = np.zeros(total_diff_features)
        for i in range(len(new)):
            y[new[i][0] : new[i][1]] = 1
        all_res.append(y)

        plt.plot(x, y)
        plt.scatter(gt[video_name][0], 0, c="#1f33b4")
        plt.scatter(gt[video_name][1], 0, c="#1f33b4")
        plt.savefig(f"features_local/{video_name}_res_{index}.png")
        plt.clf()

    upper = total_diff_features
    x = np.arange(upper)
    y = np.zeros(total_diff_features)
    for i in range(upper):
        unit = False
        for res in all_res:
            unit = unit or (res[i] != 0)
        if unit:
            y[i] = 1

    plt.plot(x, y)
    plt.scatter(gt[video_name][0], 0, c="#1f33b4")
    plt.scatter(gt[video_name][1], 0, c="#1f33b4")
    plt.savefig(f"features_local/{video_name}_res.png")
    plt.clf()
    y = np.append(y, 0)

    recall_rate, precision_rate = evaluation(y, gt, video_name, total_imgs)

    txt += "================================\n"
    txt += f"{video_name} result: \n"
    txt += f"Recall rate: {recall_rate}\n"
    txt += f"Precision rate: {precision_rate}\n"
txt += "================================\n"

with open("res.txt", "w") as f:
    f.write(txt)