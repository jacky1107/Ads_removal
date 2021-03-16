import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import torch
from torch.nn import functional as F


def lower(new):
    new = 1 - new
    return new


def lower_sigmoid(new):
    new = 1 - new
    return 1 / (1 + np.exp(-new))


def higher(new):
    return new


def higher_sigmoid(new):
    return 1 / (1 + np.exp(-new))


def reshape_segmentation(segment):
    new = []
    f = segment[0]
    for i in range(1, len(segment)):
        s = segment[i]
        new.append([f, s])
        f = s
    new = np.array(new)
    return new


def save_video(video_name, new, final_result, thres):
    resolution = {
        "test_5": (240, 352, 3),
        "test_6": (240, 352, 3),
        "Video_1": (480, 720, 3),
    }
    i, j = 0, 0
    f1, f2 = new[j][0], new[j][1]
    cap = cv2.VideoCapture(f"videos/{video_name}.avi")
    name = f"features_local/{video_name}/result/final_result.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    w = int(cap.get(3))
    h = int(cap.get(4))
    out = cv2.VideoWriter(name, fourcc, 30.0, (w, h))

    index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if final_result[i] > thres:
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            fontScale = 1.5
            color = (66, 142, 255)
            thickness = 2
            cv2.putText(
                frame,
                f"Ad_{j}",
                org,
                font,
                fontScale,
                color,
                thickness,
                cv2.LINE_AA,
            )

        out.write(frame)
        if i >= f2:
            j += 1
            if j >= len(new):
                break
            f1, f2 = new[j][0], new[j][1]
        i += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def get_segmentation_v2(diff_features, thres):
    results = []
    for i in range(len(diff_features)):
        if diff_features[i] > thres:
            results.append(i + 1)
    results = np.array(results)
    return results


def get_segmentation(diff_features, f_index_thres):
    results = []
    index = f_index_thres[0]
    thres = f_index_thres[1]
    for i in range(len(diff_features)):
        if diff_features[i, index] > thres:
            results.append(i + 1)
    results = np.array(results)
    return results


def merge_small_seg_v2(segments, thres=15):
    new = []
    i, j = 0, 0
    while i < len(segments):
        s1, s2 = segments[i][0], segments[i][1]
        if (s2 - s1) < thres:
            mean = (s1 + s2) // 2
            new[j - 1][1] = mean
            if i + 1 >= len(segments):
                break
            segments[i + 1][0] = mean
        else:
            new.append([s1, s2])
            j += 1
        i += 1
    return new


def merge_small_seg(segments, thres=15):
    new = []
    i, j = 0, 0
    while i < len(segments):
        f = segments[i][0]
        s = segments[i][1]
        if s - f < thres:
            if i == 0:
                s = segments[i + 1][1]
                i += 1
            elif i == len(segments) - 1:
                new[j - 1][1] = segments[i][1]
            else:
                new[j - 1][1] = segments[i + 1][1]
                i += 1
        else:
            new.append([f, s])
            j += 1
        i += 1
    return new


def merge_gap_between_seg(segments, thres=15):
    i, j = 0, 0
    while i < len(segments) - 1:
        f = segments[i][1]
        s = segments[i + 1][0]
        if s - f < thres:
            segments[i][1] = segments[i + 1][1]
            segments = np.delete(segments, i + 1, 0)
            i -= 1
        i += 1
    return segments


def cal_recall(confuse_matrix):
    tp = confuse_matrix[0][0]
    fn = confuse_matrix[1][0]
    return tp / (tp + fn)


def cal_precision(confuse_matrix):
    tp = confuse_matrix[0][0]
    fp = confuse_matrix[0][1]
    return tp / (tp + fp)


def evaluation(res, gt, video_name, total_imgs):
    gt_f, gt_s = gt[video_name]
    tar = np.zeros(total_imgs)
    tar[gt_f:gt_s] = 1

    confuse_matrix = [[0, 0], [0, 0]]
    for i in range(total_imgs):
        if res[i] and (gt_f <= i and i < gt_s):
            confuse_matrix[0][0] += 1
        elif res[i] and (i < gt_f or gt_s <= i):
            confuse_matrix[0][1] += 1
        elif not res[i] and (gt_f <= i and i < gt_s):
            confuse_matrix[1][0] += 1
        elif not res[i] and (i < gt_f or gt_s <= i):
            confuse_matrix[1][1] += 1

    print(confuse_matrix)
    recall_rate = cal_recall(confuse_matrix)
    precision_rate = cal_precision(confuse_matrix)

    return recall_rate, precision_rate


def evaluation_v2(res, gt, video_name, thres):
    gt_f, gt_s = gt[video_name]
    total = len(res)
    confuse_matrix = [[0, 0], [0, 0]]
    for i in range(total):
        if res[i] > thres and (gt_f <= i and i < gt_s):
            confuse_matrix[0][0] += 1
        elif res[i] > thres and (i < gt_f or gt_s <= i):
            confuse_matrix[0][1] += 1
        elif res[i] <= thres and (gt_f <= i and i < gt_s):
            confuse_matrix[1][0] += 1
        elif res[i] <= thres and (i < gt_f or gt_s <= i):
            confuse_matrix[1][1] += 1

    print(confuse_matrix)
    recall_rate = cal_recall(confuse_matrix)
    precision_rate = cal_precision(confuse_matrix)

    return recall_rate, precision_rate


def append_first_last_frame(diff_features, segment):
    seg1 = np.array([0, segment[0][0]])
    seg2 = np.array([segment[-1][1], len(diff_features)])
    segment = np.insert(segment, 0, seg1, axis=0)
    segment = np.insert(segment, len(segment), seg2, axis=0)
    return segment


def cal_seg_features_v2(diff_features, segment):
    new_features = np.zeros(len(diff_features))
    for i in range(len(segment)):
        s1, s2 = segment[i][0], segment[i][1]
        new_features[s1:s2] = np.mean(diff_features[s1:s2])
    new_features = np.array(new_features)
    return new_features


def cal_seg_features(diff_features, index, segment):
    new_features = []
    for i in range(len(segment)):
        s1, s2 = segment[i][0], segment[i][1]
        mean = np.mean(diff_features[s1:s2, index])
        new_features.append(mean)
    new_features = np.array(new_features)
    new_features = min_max_scaler(new_features)
    return new_features


def min_max_scaler(features):
    f_min = np.min(features)
    f_max = np.max(features)
    features = (features - f_min) / (f_max - f_min)
    return features


def cal_seg_features_eval(diff_features, seg_info, segment):
    new_features = []
    seg1 = np.array([0, segment[0][0] - 1])
    seg2 = np.array([segment[-1][1] + 1, len(diff_features)])
    segment = np.insert(segment, 0, seg1, axis=0)
    segment = np.insert(segment, len(segment), seg2, axis=0)
    for i in range(len(segment)):
        s1, s2 = segment[i][0], segment[i][1]
        mean = np.mean(diff_features[s1:s2, seg_info[0]])
        new_features.append(mean)

    new_features = np.array(new_features)
    new_features = min_max_scaler(new_features)
    return new_features, segment


def min_max_scaler(features):
    f_min = np.min(features)
    f_max = np.max(features)
    features = (features - f_min) / (f_max - f_min)
    return features


def cal_seg_features_eval(diff_features, seg_info, segment):
    new_features = []
    seg1 = np.array([0, segment[0][0] - 1])
    seg2 = np.array([segment[-1][1] + 1, len(diff_features)])
    segment = np.insert(segment, 0, seg1, axis=0)
    segment = np.insert(segment, len(segment), seg2, axis=0)
    for i in range(len(segment)):
        s1, s2 = segment[i][0], segment[i][1]
        mean = np.mean(diff_features[s1:s2, seg_info[0]])
        new_features.append(mean)

    new_features = np.array(new_features)
    new_features = min_max_scaler(new_features)
    return new_features, segment


def lower_is_ads(value, thres):
    return value < thres


def greater_is_ads(value, thres):
    return value > thres

    # font = cv2.FONT_HERSHEY_SIMPLEX
    # org = (50, 50)
    # fontScale = 1
    # color = (255, 0, 0)
    # thickness = 2

    # if features[i]:
    #     cv2.putText(
    #         frame, "Ad", (50, 50), font, fontScale, color, thickness, cv2.LINE_AA
    #     )
