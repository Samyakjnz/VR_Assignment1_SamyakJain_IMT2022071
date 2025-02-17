# VR_Assignment1_SamyakJain_IMT2022071
# Coin Detection and Panorama Stitching Project

This project involves two main tasks:

1. **Coin Detection and Segmentation**: Detecting and counting scattered Indian coins in images using computer vision techniques.
2. **Panorama Stitching**: Stitching multiple overlapping images into a single panoramic image using key points detection and image alignment.


### Part 1: Coin Detection and Segmentation

In Part 1, we work with an image containing scattered Indian coins. This image serves as the input data for detecting and segmenting the coins.

- **Input Image**: The image used for coin detection is a simple photograph of scattered Indian coins.
- **Image Filename**: `coinsIMAGE.jpg`
- **Image Description**: The image is taken in a controlled environment with good lighting, ensuring clear visibility of the coins. The coins are scattered in different orientations and partially overlapping, which adds a challenge for accurate detection.

### Example Input Image

Here’s the input image that is used in Part 1:

![Original image](coinsIMAGE.jpg)



#### 1. Canny Edge Detection
This image shows the edges detected using the Canny algorithm. The edges are the initial step in identifying the coins.

![Canny Edge Detection](canny.png)

#### 2. Contours of Detected Coins
This image shows the contours drawn around each detected coin. The coins are highlighted with green lines.

![Coin Contours](contours.png)

#### 3. Segmented Coins
This image shows each coin individually isolated after segmentation.

![Segmented Coins](segmented_coin_1.jpg)
#### 4. Coins Detection

This shows the correct number of coins being detected

![Detected Coins](detect'.png)
---

## Part 2: Panorama Stitching

In this part, multiple overlapping images are stitched together into a single panoramic image. Key points are extracted from the images, matched, and used to compute a homography matrix to align the images. The resulting panorama is cropped to remove black borders.

### Steps:
1. **Extract Key Points**: Key points from overlapping images are detected using the ORB detector.
2. **Image Stitching**: The images are aligned and stitched together using the matched key points and homography transformation.
3. **Cropping the Final Panorama**: Any black borders on the final stitched image are cropped to provide a clean output.

### Visual Output for Part 2

#### 1. Key Points Detection
This image shows the key points detected in the overlapping images using the ORB detector.

![Key Points Detection](key1.png)
![Key Points Detection](key2.png)
![Key Points Detection](key3.png)
#### 2. Final Stitched Panorama
This is the final stitched panorama image, showing the result of aligning and stitching the images together.

![Final Panorama](cropped_stitched_panorama.jpg)

---

## Installation

To run this project, you need Python 3.x and the required dependencies.

### Step 1: Clone the repository
```bash
git clone https://github.com/yourusername/coin-detection-panorama.git
cd coin-detection-panorama
