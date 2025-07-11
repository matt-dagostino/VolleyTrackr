import cv2
import numpy as np
import matplotlib.pyplot as plt

# List to store clicked points
points = []

# Read the image
image = cv2.imread('input_videos/input_picture.PNG')  # Change to your image filename
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib

# Show image
plt.imshow(image)
plt.title("Click on 4 corners of the court")
plt.axis("on")

# Let user click 4 points
points = plt.ginput(4, timeout=0)  # Wait until 4 clicks
plt.close()

# Print selected points
print("Selected Points:", points)

# Define the court boundaries manually as a polygon
court_boundary = np.array(points, np.int32)  # Example coordinates
court_boundary = court_boundary.reshape((-1, 1, 2))

# Load the reference image
frame = cv2.imread('input_videos/input_picture.PNG')

# Draw the court boundary on the image for visualization
cv2.polylines(frame, [court_boundary], isClosed=True, color=(0, 255, 0), thickness=5)

# Show the image
cv2.imwrite("court_boundary_output.jpg", frame)  # Save the image
