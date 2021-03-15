import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def get_segmentation(diff_features, index, thres):
    results = []
    for i in range(len(diff_features)):
        if diff_features[i, index] > thres:
            results.append(i + 1)
    results = np.array(results)
    return results


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
        elif res[i] and (i < gt_f or gt_s >= i):
            confuse_matrix[0][1] += 1
        elif not res[i] and (gt_f <= i and i < gt_s):
            confuse_matrix[1][0] += 1
        elif not res[i] and (i < gt_f or gt_s >= i):
            confuse_matrix[1][1] += 1

    recall_rate = cal_recall(confuse_matrix)
    precision_rate = cal_precision(confuse_matrix)

    return recall_rate, precision_rate
