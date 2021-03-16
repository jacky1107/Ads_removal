import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from utils import *

video_names = ["test_5", "test_6", "Video_1"]
features_index_thres = {
    "test_5": [
        (
            18,
            [
                (11, lower),
                (25, higher),
                (26, lower),
                (27, lower),
                (30, lower),
                (31, lower),
            ],
        ),
        (
            19,
            [
                (3, higher),
            ],
        ),
    ],
    "test_6": [
        (
            18,
            [
                (11, higher),
            ],
        ),
        (
            19,
            [
                (11, higher),
            ],
        ),
    ],
    "Video_1": [
        (
            18,
            [
                (8, lower),
                (26, lower_sigmoid),
                (24, higher),
                (22, higher),
            ],
        ),
        (
            33,
            [
                (4, higher),
                (10, lower),
                (29, lower),
            ],
        ),
    ],
}

final_filter = {
    "Video_1": [0.085, 50, 0.05],
    "test_6": [0.19, 50, 0.12],
    "test_5": [0.22, 50, 0.15],
}

gt = {
    "test_5": [2800, 7870],
    "test_6": [2776, 8320],
    "Video_1": [939, 5438],
}

total_recall = 0
total_precision = 0

txt = ""
for video_name in video_names:
    final_results = []

    flag = False
    for f_index_thres in features_index_thres[video_name]:
        seg_index = f_index_thres[0]
        file = f"new_{video_name}_{seg_index}.npy"
        if not os.path.isfile(file):
            break
        with open(file, "rb") as f:
            features = np.load(f, allow_pickle=True)
        print(features.shape)
        f_num, total = features.shape

        if len(f_index_thres[1]) == 0:
            break

        flag = True
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
        plt.scatter(gt[video_name][0], 0, c="#1f33b4")
        plt.scatter(gt[video_name][1], 0, c="#1f33b4")
        plt.savefig(f"features/{video_name}/segment/fuse_{seg_index}.png")
        plt.clf()

    fuse = 0
    for res in final_results:
        fuse += res
    fuse = min_max_scaler(fuse)

    x = np.arange(total)
    plt.plot(x, fuse)
    plt.scatter(gt[video_name][0], 0, c="#1f33b4")
    plt.scatter(gt[video_name][1], 0, c="#1f33b4")
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
    print(seg)

    x = np.arange(total)
    plt.plot(x, diff)
    plt.scatter(gt[video_name][0], 0, c="#1f33b4")
    plt.scatter(gt[video_name][1], 0, c="#1f33b4")
    plt.savefig(f"features/{video_name}/diff.png")
    plt.clf()

    new_features = cal_seg_features_v2(fuse, seg)
    x = np.arange(total)
    plt.plot(x, new_features)
    plt.scatter(gt[video_name][0], 0, c="#1f33b4")
    plt.scatter(gt[video_name][1], 0, c="#1f33b4")
    plt.savefig(f"features/{video_name}/final_result.png")
    plt.clf()

    # Evaluation
    final_result = np.zeros(len(new_features) + 1)
    final_result[: len(new_features)] = new_features
    print(final_result.shape)

    save_video(video_name, seg, final_result, final_filter[video_name][-1])

    recall_rate, precision_rate = evaluation_v2(
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
