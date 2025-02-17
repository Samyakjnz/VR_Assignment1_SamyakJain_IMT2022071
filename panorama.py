import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the images
image_paths = ['p1.jpg', 'p2.jpg', 'p3.jpg']  # Replace with your actual image file paths
images = [cv2.imread(img_path) for img_path in image_paths]

# Initialize the ORB detector
orb = cv2.ORB_create()

# Detect keypoints and descriptors for each image
keypoints_list = []
descriptors_list = []

for img in images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    keypoints_list.append(keypoints)
    descriptors_list.append(descriptors)

# Step 1: Detect keypoints and visualize them
# Visualize keypoints for each image
for i, img in enumerate(images):
    img_with_keypoints = cv2.drawKeypoints(img, keypoints_list[i], None, color=(0, 255, 0), flags=0)
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"Image {i + 1} - Keypoints")
    plt.show()

# Create a brute force matcher to match keypoints between images
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors between the first two images
matches_1_2 = bf.match(descriptors_list[0], descriptors_list[1])

# Sort the matches based on distance (best matches first)
matches_1_2 = sorted(matches_1_2, key=lambda x: x.distance)

# Get the points from the matches
pts1 = np.float32([keypoints_list[0][m.queryIdx].pt for m in matches_1_2]).reshape(-1, 1, 2)
pts2 = np.float32([keypoints_list[1][m.trainIdx].pt for m in matches_1_2]).reshape(-1, 1, 2)

# Compute the homography matrix
H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

# Warp the second image to align it with the first image
height, width = images[0].shape[:2]
result_1_2 = cv2.warpPerspective(images[1], H, (width + images[1].shape[1], height))

# Place the first image on the left side of the result
result_1_2[0:height, 0:width] = images[0]

# Now we need to stitch the third image (if applicable)
matches_2_3 = bf.match(descriptors_list[1], descriptors_list[2])
matches_2_3 = sorted(matches_2_3, key=lambda x: x.distance)

pts1 = np.float32([keypoints_list[1][m.queryIdx].pt for m in matches_2_3]).reshape(-1, 1, 2)
pts2 = np.float32([keypoints_list[2][m.trainIdx].pt for m in matches_2_3]).reshape(-1, 1, 2)

H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

# Warp the third image to align with the second image
height, width = result_1_2.shape[:2]
result_1_2_3 = cv2.warpPerspective(images[2], H, (width + images[2].shape[1], height))

# Place the result of the first two images into the stitched result
result_1_2_3[0:height, 0:width] = result_1_2

# Step 2: Remove black borders by cropping the image
# Convert the result image to grayscale to detect non-black regions
gray_result = cv2.cvtColor(result_1_2_3, cv2.COLOR_BGR2GRAY)

# Apply a threshold to create a binary image (black and white)
_, thresholded = cv2.threshold(gray_result, 1, 255, cv2.THRESH_BINARY)

# Find contours in the thresholded image
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get the bounding box of the largest contour (the stitched area)
x, y, w, h = cv2.boundingRect(contours[0])

# Crop the image to the bounding box of the non-black regions
cropped_result = result_1_2_3[y:y+h, x:x+w]

# Step 3: Show the final cropped stitched panorama
plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(cropped_result, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Cropped Stitched Panorama")
plt.show()

# Optionally, save the cropped panorama
cv2.imwrite("cropped_stitched_panorama.jpg", cropped_result)
