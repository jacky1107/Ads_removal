import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from utils import *

save = True
video_names = ["test_5", "test_6", "Video_1"]
features_index_thres = {
    "test_5": [
        (18, 0.24, 5),
        (19, 0.15, 5),
    ],
    "test_6": [
        (18, 0.165, 5),
        (19, 0.0885, 5),
    ],
    "Video_1": [
        (18, 0.25, 5),
        (33, 0.00027, 20),
    ],
}
gt = {
    "test_5": [2800, 7870],
    "test_6": [2776, 8320],
    "Video_1": [939, 5438],
}

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

    for f_index_thres in features_index_thres[video_name]:
        all_res = []
        seg_index = f_index_thres[0]
        new = get_segmentation(diff_features, f_index_thres)
        new = reshape_segmentation(new)
        new = append_first_last_frame(diff_features, new)
        new = merge_small_seg_v2(new, f_index_thres[2])
        print(new)
        for index in range(features.shape[1]):
            new_features = cal_seg_features(diff_features, index, new)
            x = np.arange(total_diff_features)
            y = np.zeros(total_diff_features)
            for i in range(len(new)):
                y[new[i][0] : new[i][1]] = new_features[i]

            if save:
                plt.plot(x, y)
                plt.scatter(gt[video_name][0], 0, c="#1f33b4")
                plt.scatter(gt[video_name][1], 0, c="#1f33b4")
                plt.savefig(
                    f"features/{video_name}/segment/test_{seg_index}_{index}.png"
                )
                plt.clf()

            all_res.append(y)

        all_res = np.array(all_res, dtype=object)
        print(all_res.shape)
        with open(f"new_{video_name}_{seg_index}.npy", "wb") as f:
            np.save(f, all_res)

        x = np.arange(total_diff_features)
        y = np.zeros(total_diff_features)
        for i in range(len(new)):
            y[new[i][0] : new[i][1]] = i

        if save:
            plt.plot(x, y)
            plt.scatter(gt[video_name][0], 0, c="#1f33b4")
            plt.scatter(gt[video_name][1], 0, c="#1f33b4")
            plt.savefig(f"features/{video_name}/segment/res_{seg_index}.png")
            plt.clf()
