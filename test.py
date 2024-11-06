import numpy as np
import cv2

# Define the image size
image_height, image_width = 1380, 876

# Initialize a blank mask
mask = np.zeros((image_height, image_width), dtype=np.uint8)

pp = [
    {
        "x": 284.25886143931257,
        "y": 666.72126745435014
    },
    {
        "x": 200,
        "y": 735
    },
    {
        "x": 157,
        "y": 812
    },
    {
        "x": 113,
        "y": 924
    },
    {
        "x": 230,
        "y": 1056
    },
    {
        "x": 472,
        "y": 1169
    },
    {
        "x": 575,
        "y": 1190
    },
    {
        "x": 653,
        "y": 1171
    },
    {
        "x": 798,
        "y": 965
    },
    {
        "x": 812,
        "y": 795
    },
    {
        "x": 742,
        "y": 667
    },
    {
        "x": 624,
        "y": 621
    },
    {
        "x": 512,
        "y": 644
    },
    {
        "x": 381,
        "y": 634
    }
]
points = []
for i in pp:
    points.append((i["x"], i["y"]))
print(points)
points = np.array(points, np.int32)

# Define your points as a list of (x, y) coordinates
# points = np.array([[100, 100], [200, 80], [250, 200], [150, 250], [90, 180]], np.int32)
points = points.reshape((-1, 1, 2))

# Draw the filled polygon on the mask
k = cv2.fillPoly(mask, [points], color=255)
print(k)

# Display the mask (if you have matplotlib)
import matplotlib.pyplot as plt

plt.imshow(mask, cmap='gray')
plt.show()
