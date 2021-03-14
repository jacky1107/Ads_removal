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


gt = {"test_5": [2800, 7870], "test_6": [2776, 8320], "Video_1": [939, 5438]}
video_name = "test_6"
with open("features.npy", "rb") as f:
    features = np.load(f, allow_pickle=True)
print(features.shape)

total_imgs = len(features)
diff_features = ((features[: total_imgs - 1, :] - features[1:, :]) ** 2) ** 0.5
total_diff_features = len(diff_features)


# diff
total_diff_features = len(diff_features)
for i in range(len(diff_features[0])):
    upper = total_diff_features - 1
    x = np.arange(0, upper, upper / total_diff_features)
    plt.plot(x, diff_features[:, i])
    plt.scatter(gt[video_name][0], 0, c="#1f33b4")
    plt.scatter(gt[video_name][1], 0, c="#1f33b4")
    plt.savefig(f"features/{video_name}_{i}_diff.png")
    plt.clf()

# origin
for i in range(len(features[0])):
    x = np.arange(total_imgs)
    plt.plot(x, features[:, i])
    plt.scatter(gt[video_name][0], 0, c="#1f33b4")
    plt.scatter(gt[video_name][1], 0, c="#1f33b4")
    plt.savefig(f"features/{video_name}_{i}.png")
    plt.clf()


new = get_segmentation(diff_features, 13, 280)
print(new)
assert len(new) % 2 == 0
new = new.reshape((len(new) // 2, 2))
for i in range(5):
    new = merge_small_seg(new, 30)
    new = merge_gap_between_seg(new, 30)
print(new)

upper = total_diff_features - 1
x = np.arange(upper)
y = np.zeros(total_diff_features - 1)
for i in range(len(new)):
    y[new[i][0] : new[i][1]] = 50

plt.plot(x, y)
plt.scatter(gt[video_name][0], 0, c="#1f33b4")
plt.scatter(gt[video_name][1], 0, c="#1f33b4")
plt.savefig(f"features/{video_name}_res.png")
plt.clf()
