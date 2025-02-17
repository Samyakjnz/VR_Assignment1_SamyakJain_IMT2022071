import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_coins(image_path):
    """
    Detects coins in the image using edge detection and contours.
    Draws contours around detected coins and displays the image.
    """
    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not read image.")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise and make edge detection more reliable
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # Perform edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Print the number of contours detected (this is the number of coins detected)
    print(f"Number of coins detected: {len(contours)}")

    # Create a copy of the original image to draw contours on
    output_image = image.copy()

    # Draw contours on the original image (in green)
    cv2.drawContours(output_image, contours, -1, (0, 255, 0), 3)

    # Convert BGR to RGB for displaying with Matplotlib
    output_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    # Visualize the Canny edge-detected image
    plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(edges, cmap='gray')
    plt.title("Canny Edge Detection")
    plt.axis("off")

    # Visualize the original image with contours drawn around detected coins
    plt.subplot(1, 2, 2)
    plt.imshow(output_rgb)
    plt.title("Detected Coins (Green Contours)")
    plt.axis("off")

    plt.show()

    return contours, image, edges

def segment_coins(contours, image, edges):
    """
    Segments individual coins from the image based on the contours.
    Displays and saves the segmented coin images.
    """
    coin_images = []
    for idx, contour in enumerate(contours):
        # Create a mask for the current coin
        mask = np.zeros(edges.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Apply the mask to extract the coin from the original image
        segmented_coin = cv2.bitwise_and(image, image, mask=mask)

        # Extract the bounding box for the contour
        x, y, w, h = cv2.boundingRect(contour)

        # To ensure the entire coin is captured, we slightly adjust the bounding box
        # Expanding the bounding box to include more area around the coin
        padding = 5  # Adjust padding to ensure full capture of the coin
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)

        # Crop the image around the bounding box of the coin
        coin_cropped = segmented_coin[y:y+h, x:x+w]

        # Add the cropped coin image to the list
        coin_images.append(coin_cropped)

        # Display each segmented coin
        plt.figure(figsize=(4, 4))
        plt.imshow(cv2.cvtColor(coin_cropped, cv2.COLOR_BGR2RGB))
        plt.title(f"Segmented Coin {idx + 1}")
        plt.axis("off")
        plt.show()

        # Optionally, save the segmented coin images
        cv2.imwrite(f"segmented_coin_{idx + 1}.jpg", coin_cropped)

    return coin_images


# Main Execution
if __name__ == "__main__":
    image_path = "coinsIMAGE.jpg"  # Change the path to your image file

    # Part A: Detect coins in the image
    contours, image, edges = detect_coins(image_path)

    if contours is not None:
        # Part B: Segment individual coins
        segment_coins(contours, image, edges)
