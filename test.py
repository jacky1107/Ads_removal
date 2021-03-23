import cv2
import numpy as np

path = "/home/jacky/Documents/code/Ads_removal/videos/TVshow_7.avi"
cap = cv2.VideoCapture(path)
j = 0
features = []
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    feature = []
    mean, std = cv2.meanStdDev(img)
    mean = np.mean(mean)
    for i in range(38):
        feature.append(mean)
    features.append(feature)
    j += 1
    print(j)

features = np.array(features, dtype=object)
print(features.shape)
with open(f"test_features.npy", "wb") as f:
    np.save(f, features)

cap.release()
cv2.destroyAllWindows()