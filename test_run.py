import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the reference image
image = cv2.imread('input_videos/input_picture.PNG')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Show image and let user select 4 points
plt.imshow(image)
plt.title("Click on 4 corners of the court")
plt.axis("on")

# Get user input for 4 corners
points = plt.ginput(4, timeout=0)
plt.close()

# Convert points to NumPy array
court_boundary = np.array(points, np.int32).reshape((-1, 1, 2))

# Draw the court boundary
cv2.polylines(image, [court_boundary], isClosed=True, color=(0, 255, 0), thickness=5)

# Save the image
cv2.imwrite("court_boundary_output.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

print("Court boundary saved as 'court_boundary_output.jpg'")


from ultralytics import YOLO

# Load YOLO model
model = YOLO('models/best.pt')

# Run detection on video
results = model.predict('input_videos/input_video.webm', conf=0.2, save=True)

# Function to check if the ball is inside the court
def is_inside_court(point, court_boundary):
    return cv2.pointPolygonTest(court_boundary, point, False) >= 0

# Iterate over detected objects
for result in results:
    for box in result.boxes:
        x_center = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
        y_center = int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)
        ball_position = (x_center, y_center)

        if is_inside_court(ball_position, court_boundary):
            print(f"Ball is **INSIDE** the court at {ball_position}")
        else:
            print(f"Ball is **OUTSIDE** the court at {ball_position}")
