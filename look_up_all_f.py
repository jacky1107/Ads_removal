import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from utils import *

cal_seg = True
save_feature = False
save_video_avi = False
video_names = ["Video_1"]

txt = ""
for video_name in video_names:
    gt = {"test_5": [2800, 7870], "test_6": [2776, 8320], "Video_1": [939, 5438]}

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

    features_index_thres = {
        "test_5": [
            (diff_features, 18, 0.24, 5),
            (diff_features, 19, 0.15, 5),
        ],
        "test_6": [
            (diff_features, 18, 0.18, 5),
            (diff_features, 19, 0.09, 5),
        ],
        "Video_1": [
            # (diff_features, 18, 0.25, 5),
            # (diff_features, 19, 0.101, 5),
            (diff_features, 33, 0.00027, 5),
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
            plt.savefig(f"features_local/{video_name}/{i}_diff.png")
            plt.clf()

        # origin
        for i in range(len(features[0])):
            x = np.arange(total_imgs)
            plt.plot(x, features[:, i])
            plt.scatter(gt[video_name][0], 0, c="#1f33b4")
            plt.scatter(gt[video_name][1], 0, c="#1f33b4")
            plt.savefig(f"features_local/{video_name}/{i}.png")
            plt.clf()

    if cal_seg:
        all_res = []
        for f_index_thres in features_index_thres[video_name]:
            new = get_segmentation(
                f_index_thres[0],
                f_index_thres[1],
                f_index_thres[2],
            )
            print(new)
            new = reshape_segmentation(new)
            new = merge_small_seg(new, f_index_thres[3])
            # new = merge_gap_between_seg(new, f_index_thres[3])
            print(new)

            seg_index = f_index_thres[1]
            # =====
            for index in range(features.shape[1]):
                new_features, new_segments = cal_seg_features(f_index_thres, index, new)
                upper = total_diff_features
                x = np.arange(upper)
                y = np.zeros(total_diff_features)
                for i in range(len(new_segments)):
                    if new_features[i] > 0.7:
                        new_features[i] = 0
                    y[new_segments[i][0] : new_segments[i][1]] = new_features[i]
                plt.plot(x, y)
                plt.scatter(gt[video_name][0], 0, c="#1f33b4")
                plt.scatter(gt[video_name][1], 0, c="#1f33b4")
                plt.savefig(f"features_local/{video_name}/test_{seg_index}_{index}.png")
                plt.clf()
            # =====

            upper = total_diff_features
            x = np.arange(upper)
            y = np.zeros(total_diff_features)
            for i in range(len(new)):
                y[new[i][0] : new[i][1]] = 1
            all_res.append(y)

            plt.plot(x, y)
            plt.scatter(gt[video_name][0], 0, c="#1f33b4")
            plt.scatter(gt[video_name][1], 0, c="#1f33b4")
            plt.savefig(f"features_local/{video_name}/res_{seg_index}.png")
            plt.clf()
