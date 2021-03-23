import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from utils import *

save = False
video_names = ["TVshow_7"]
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
    "TVshow_7": [
        (0, 5 * 1e-4, 5),
        (1, 15 * 1e-4, 5),
        (8, 0.3 * 1e-10, 5),
        (9, 0.5 * 1e-9, 5),
        (10, 0.8 * 1e-10, 5),
        (20, 0.003, 5),
        (23, 0.0007, 5),
        (24, 0.003, 5),
        (25, 0.00067, 5),
    ],
}
gt = {
    "test_5": [2800, 7870],
    "test_6": [2776, 8320],
    "Video_1": [939, 5438],
    "TVshow_7": [[14389, 18143], [28133, 31893]],
}
features_index_thres_final = {
    "TVshow_7": [
        (
            1,
            [
                (4, higher),
            ],
        ),
        (
            24,
            [
                (5, higher),
            ],
        ),
    ],
}

final_filter = {"TVshow_7": [0.1, 50, 0.15]}
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
    for i in range(total_diff_features):
        for j in range(len(diff_features[i])):
            if diff_features[i, j] == np.nan:
                diff_features[i, j] = 0

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
                if new_features[i] is np.nan:
                    y[new[i][0]: new[i][1]] = 0
                else:
                    y[new[i][0]: new[i][1]] = new_features[i]

            if save:
                plt.plot(x, y)
                for segs in gt[video_name]:
                    for xs in segs:
                        plt.axvline(x=xs, color="r", linestyle=":")
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
            y[new[i][0]: new[i][1]] = i

        if save:
            plt.plot(x, y)
            for segs in gt[video_name]:
                for x in segs:
                    plt.axvline(x=x, color="r", linestyle=":")
            plt.savefig(f"features/{video_name}/segment/res_{seg_index}.png")
            plt.clf()

total_recall = 0
total_precision = 0

txt = ""
for video_name in video_names:
    final_results = []

    for f_index_thres in features_index_thres_final[video_name]:
        seg_index = f_index_thres[0]
        print(seg_index)
        file = f"new_{video_name}_{seg_index}.npy"
        if not os.path.isfile(file):
            break
        with open(file, "rb") as f:
            features = np.load(f, allow_pickle=True)
        f_num, total = features.shape

        if len(f_index_thres[1]) == 0:
            break

        fuse = 1
        for index, func in f_index_thres[1]:
            feature = features[index]
            feature = np.array(feature, dtype=float)
            feature = func(feature)
            feature = min_max_scaler(feature)
            fuse *= feature
        fuse = min_max_scaler(fuse)
        final_results.append(fuse)

        x = np.arange(total)
        plt.plot(x, fuse)
        for segs in gt[video_name]:
            for x in segs:
                plt.axvline(x=x, color="r", linestyle=":")
        plt.savefig(f"features/{video_name}/segment/fuse_{seg_index}.png")
        plt.clf()

    fuse = 0
    for res in final_results:
        fuse += res
    fuse = min_max_scaler(fuse)

    x = np.arange(total)
    plt.plot(x, fuse)
    for segs in gt[video_name]:
        for x in segs:
            plt.axvline(x=x, color="r", linestyle=":")
    plt.savefig(f"features/{video_name}/fuse.png")
    plt.clf()

    # filter
    diff = []
    for i in range(1, total):
        dist = ((fuse[i] - fuse[i - 1]) ** 2) ** 0.5
        diff.append(dist)
    diff.append(diff[-1])
    diff = np.array(diff)
    seg = get_segmentation_v2(diff, final_filter[video_name][0])
    seg = reshape_segmentation(seg)
    seg = append_first_last_frame(diff, seg)
    seg = merge_small_seg_v2(seg, final_filter[video_name][1])

    x = np.arange(total)
    plt.plot(x, diff)
    for segs in gt[video_name]:
        for x in segs:
            plt.axvline(x=x, color="r", linestyle=":")
    plt.savefig(f"features/{video_name}/diff.png")
    plt.clf()

    new_features = cal_seg_features_v2(fuse, seg)
    x = np.arange(total)
    plt.plot(x, new_features)
    for segs in gt[video_name]:
        for x in segs:
            plt.axvline(x=x, color="r", linestyle=":")
    plt.savefig(f"features/{video_name}/final_result.png")
    plt.clf()

    # Evaluation
    final_result = np.zeros(len(new_features) + 1)
    final_result[: len(new_features)] = new_features

    # save_video(video_name, seg, final_result, final_filter[video_name][-1])

    recall_rate, precision_rate = evaluation_v3(
        final_result, gt, video_name, final_filter[video_name][-1]
    )

    total_recall += recall_rate
    total_precision += precision_rate

    txt += "================================\n"
    txt += f"{video_name} result: \n"
    txt += f"Recall rate: {recall_rate}\n"
    txt += f"Precision rate: {precision_rate}\n"
txt += "================================\n"

print(total_recall / len(video_names))
print(total_precision / len(video_names))

with open("res.txt", "w") as f:
    f.write(txt)
