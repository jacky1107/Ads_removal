import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from torch import nn
from torchvision.utils import save_image


def cal_mean_std(img):
    mean, std = cv2.meanStdDev(img)
    std = np.mean(std)
    mean = np.mean(mean)
    return mean, std


def cal_high_freq(img):
    assert len(img.shape) == 2, "Expect gray scale image"
    h, w = img.shape
    tensor_img = torch.Tensor(img)
    tensor_img = tensor_img.view(1, 1, h, w)
    tensor_max_img = max_pool(tensor_img)
    tensor_avg_img = avg_pool(tensor_img)
    diff = abs(tensor_avg_img - tensor_max_img)
    diff = (diff - torch.min(diff)) / torch.max(diff)
    diff = diff[:, :, :h, :w]
    diff = diff.view(h, w, 1)
    max_avg_diff_img = diff.numpy()
    return max_avg_diff_img


def normalized_image(img):
    for i in range(3):
        img = (img - normalized_mean[i]) / normalized_std[i]
    return img


video_name = "Video_1"
print(video_name)

normalized_std = np.array([0.229, 0.224, 0.225])
normalized_mean = np.array([0.485, 0.456, 0.406])
gamma = 1.5
max_pool = nn.MaxPool2d((2, 2), 1, padding=1)
avg_pool = nn.AvgPool2d((2, 2), 1, padding=1)

count = 0
features = []
total = {"test_5": 9693, "test_6": 10974, "Video_1": 10250}
cap = cv2.VideoCapture(f"videos/{video_name}.avi")
while True:
    res, img = cap.read()
    if not res:
        break

    # Color Space
    b, g, r = cv2.split(img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur - Low pass filter
    blur_img = cv2.GaussianBlur(img, (3, 3), 0)
    gray_blur_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
    gray_median_blur_img = cv2.medianBlur(gray_img, 3)

    # Detail and noise - High pass filter
    gray_mvg_img = cal_high_freq(gray_img)
    gray_blur_mvg_img = cal_high_freq(gray_blur_img)
    gray_median_blur_mvg_img = cal_high_freq(gray_median_blur_img)

    img_g_m_i = abs(img - gray_mvg_img)
    img_g_b_m_i = abs(img - gray_blur_mvg_img)
    img_g_mb_m_i = abs(img - gray_median_blur_mvg_img)

    # Enhancement
    # gamma_img = img ** (1 / gamma)
    # blur_gamma_img = blur_img ** (1 / gamma)
    # gray_blur_gamma_img = gray_blur_img ** (1 / gamma)
    # gray_median_blur_gamma_img = gray_median_blur_img ** (1 / gamma)

    gamma_img_g_m_i = img_g_m_i ** (1 / gamma)
    gamma_img_g_b_m_i = img_g_b_m_i ** (1 / gamma)
    gamma_img_g_mb_m_i = img_g_mb_m_i ** (1 / gamma)

    # gamma_img = normalized_image(gamma_img)
    # blur_gamma_img = normalized_image(blur_gamma_img)
    # gray_blur_gamma_img = normalized_image(gray_blur_gamma_img)
    # gray_median_blur_gamma_img = normalized_image(gray_median_blur_gamma_img)

    gamma_img_g_m_i = gamma_img_g_m_i ** (1 / gamma)
    gamma_img_g_b_m_i = gamma_img_g_b_m_i ** (1 / gamma)
    gamma_img_g_mb_m_i = gamma_img_g_mb_m_i ** (1 / gamma)

    low_freq = [img_g_m_i, img_g_b_m_i, img_g_mb_m_i]
    high_freq = [gray_mvg_img, gray_blur_mvg_img, gray_median_blur_mvg_img]
    enhancement = [
        gamma_img_g_m_i,
        gamma_img_g_b_m_i,
        gamma_img_g_mb_m_i,
    ]

    # low_freq = [blur_img, gray_blur_img, gray_median_blur_img]
    # high_freq = [gray_mvg_img, gray_blur_mvg_img, gray_median_blur_mvg_img]
    # enhancement = [
    #     gamma_img,
    #     blur_gamma_img,
    #     gray_blur_gamma_img,
    #     gray_median_blur_gamma_img,
    # ]

    feature = []
    all_in_one = [low_freq, high_freq, enhancement]
    for domain in all_in_one:
        for img in domain:
            mean, std = cal_mean_std(img)
            feature.append(mean)
            feature.append(std)
    features.append(feature)
    print(f"\r{round((count / total[video_name]) * 100, 2)}", end=" ")
    count += 1

features = np.array(features, dtype=object)
print(features.shape)
with open(f"{video_name}_features_20.npy", "wb") as f:
    np.save(f, features)

cap.release()
