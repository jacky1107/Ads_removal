import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch import nn

gt = {
    "test_5": [2800, 7870],
    "test_6": [2776, 8320],
    "Video_1": [939, 5438],
}

normalized_std = np.array([0.229, 0.224, 0.225])
normalized_mean = np.array([0.485, 0.456, 0.406])

c = 0
features = []
video_name = "Video_1"
cap = cv2.VideoCapture(f"videos/{video_name}.avi")
while True:
    res, img = cap.read()
    if not res:
        break

    pool = nn.MaxPool2d((2, 2))(torch.Tensor(img))

    mean, std = cv2.meanStdDev(img)
    std = np.mean(std)
    mean = np.mean(mean)

    for i in range(3):
        img = (img - normalized_mean[i]) / normalized_std[i]

    normal_mean, normal_std = cv2.meanStdDev(img)
    normal_std = np.mean(normal_std)
    normal_mean = np.mean(normal_mean)
    features.append([mean, std, normal_mean, normal_std])
    print(c)
    c += 1

features = np.array(features, dtype=object)
total_imgs = len(features)
print(features.shape)

# diff
diff_features = ((features[: total_imgs - 1, :] - features[1:, :]) ** 2) ** 0.5
total_diff_features = len(diff_features)
for i in range(len(diff_features[0])):
    upper = total_diff_features - 1
    x = np.arange(0, upper, upper / total_diff_features)
    plt.scatter(gt[video_name][0], 0, c="#1f33b4")
    plt.scatter(gt[video_name][1], 0, c="#1f33b4")
    plt.plot(x, diff_features[:, i])
    plt.savefig(f"features/{video_name}_{i}_diff.png")
    plt.clf()


# origin
print(features.shape)
for i in range(len(features[0])):
    x = np.arange(total_imgs)
    plt.scatter(gt[video_name][0], 0, c="#1f33b4")
    plt.scatter(gt[video_name][1], 0, c="#1f33b4")
    plt.plot(x, features[:, i])
    plt.savefig(f"features/{video_name}_{i}.png")
    plt.clf()

cap.release()
