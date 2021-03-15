import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from utils import *


cal_seg = True
save_feature = False
save_video_avi = False
video_names = ["test_5", "Video_1", "test_6"]  # , "Video_1"]

txt = ""
for video_name in video_names:
    gt = {"test_5": [2800, 7870], "test_6": [2776, 8320], "Video_1": [939, 5438]}
    with open(f"{video_name}_features.npy", "rb") as f:
        features = np.load(f, allow_pickle=True)
    print(features.shape)

    f_min = np.min(features)
    f_max = np.max(features)
    features = (features - f_min) / (f_max - f_min)
    total_imgs = len(features)
    diff_features = ((features[: total_imgs - 1, :] - features[1:, :]) ** 2) ** 0.5
    total_diff_features = len(diff_features)

    features_index_thres = {
        "test_5": [
            (diff_features, 18, 0.24, 5),
            (diff_features, 19, 0.15, 5),
            # (diff_features, 18, 750, 5),
            # (diff_features, 19, 500, 5),
        ],
        "test_6": [
            (diff_features, 18, 0.18, 5),
            (diff_features, 19, 0.09, 5),
        ],
        "Video_1": [
            (diff_features, 18, 0.25, 5),
            (diff_features, 19, 0.12, 5),
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

    if cal_seg:
        all_res = []
        for f_index_thres in features_index_thres[video_name]:
            index = f_index_thres[1]
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

            # =====
            for index in range(features.shape[1]):
                new_features, new_segments = cal_seg_features(f_index_thres, index, new)
                upper = total_diff_features
                x = np.arange(upper)
                y = np.zeros(total_diff_features)
                for i in range(len(new_segments)):
                    y[new_segments[i][0] : new_segments[i][1]] = new_features[i]
                plt.plot(x, y)
                plt.scatter(gt[video_name][0], 0, c="#1f33b4")
                plt.scatter(gt[video_name][1], 0, c="#1f33b4")
                plt.savefig(f"features_local/{video_name}_test_{index}.png")
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
            plt.savefig(f"features_local/{video_name}_res_{index}.png")
            plt.clf()

            if save_video_avi:
                save_video(video_name, index, y)


#         upper = total_diff_features
#         x = np.arange(upper)
#         y = np.zeros(total_diff_features)
#         for i in range(upper):
#             unit = False
#             for res in all_res:
#                 unit = unit or (res[i] != 0)
#             if unit:
#                 y[i] = 1

#         plt.plot(x, y)
#         plt.scatter(gt[video_name][0], 0, c="#1f33b4")
#         plt.scatter(gt[video_name][1], 0, c="#1f33b4")
#         plt.savefig(f"features_local/{video_name}_res.png")
#         plt.clf()
#         y = np.append(y, 0)

#         recall_rate, precision_rate = evaluation(y, gt, video_name, total_imgs)

#         txt += "================================\n"
#         txt += f"{video_name} result: \n"
#         txt += f"Recall rate: {recall_rate}\n"
#         txt += f"Precision rate: {precision_rate}\n"


# if cal_seg:
#     txt += "================================\n"
#     with open("res.txt", "w") as f:
#         f.write(txt)
