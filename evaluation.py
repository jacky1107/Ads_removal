import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from utils import *

cal_seg = True
save_feature = True
save_video_avi = False
video_names = ["test_5", "Video_1", "test_6"]
gt = {"test_5": [2800, 7870], "test_6": [2776, 8320], "Video_1": [939, 5438]}
features_index_thres = {
    "test_5": [
        {
            18: (
                0.24,
                5,
                [
                    (26, lower_is_ads, 0.145),
                    (26, greater_is_ads, 0.2),
                ],
            ),
        },
    ],
    "test_6": [
        {
            18: (
                0.161,
                5,
                [
                    (11, greater_is_ads, 0.115),
                ],
            ),
        },
        {
            19: (
                0.0885,
                5,
                [
                    (11, greater_is_ads, 0.06),
                ],
            ),
        },
    ],
    "Video_1": [
        {
            18: (
                0.25,
                5,
                [
                    (26, lower_is_ads, 0.15),
                    (32, greater_is_ads, 0.8),
                ],
            ),
        },
        {
            33: (
                0.00027,
                5,
                [
                    (29, lower_is_ads, 0.1),
                    (23, greater_is_ads, 0.08),
                    (31, lower_is_ads, 0.15),
                ],
            ),
        },
    ],
}

txt = ""
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

    all_res = []
    final_result = np.zeros(total_imgs)
    for f_dicts in features_index_thres[video_name]:
        for seg_index, f_index_thres in f_dicts.items():
            new = get_segmentation(
                diff_features,
                seg_index,
                f_index_thres[0],
            )
            new = reshape_segmentation(new)
            new = merge_small_seg(new, f_index_thres[1])
            print(new)
            for seg_info in f_index_thres[-1]:
                new_features, new_segments = cal_seg_features_eval(
                    diff_features, seg_info, new
                )
                x = np.arange(total_diff_features)
                y = np.zeros(total_diff_features)
                for i in range(len(new_segments)):
                    y[new_segments[i][0] : new_segments[i][1]] = new_features[i]

                for i in range(total_diff_features):
                    if seg_info[1](y[i], seg_info[2]):
                        final_result[i] = 1
    x = np.arange(total_imgs)
    plt.plot(x, final_result)
    plt.scatter(gt[video_name][0], 0, c="#1f33b4")
    plt.scatter(gt[video_name][1], 0, c="#1f33b4")
    plt.savefig(f"features_local/{video_name}/final_result.png")
    plt.clf()

    if save_video_avi:
        save_video(video_name, index, y)

    recall_rate, precision_rate = evaluation(final_result, gt, video_name, total_imgs)

    txt += "================================\n"
    txt += f"{video_name} result: \n"
    txt += f"Recall rate: {recall_rate}\n"
    txt += f"Precision rate: {precision_rate}\n"
txt += "================================\n"
with open("res.txt", "w") as f:
    f.write(txt)
