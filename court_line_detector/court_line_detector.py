import cv2
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class CourtLineDetector:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        # Load the reference image
        image_default = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(image_default, cv2.COLOR_BGR2RGB)
        image = self.transform(img_rgb).unsqueeze(0)

        # Show image and let user select 4 points
        plt.imshow(img_rgb)
        plt.title("Click on 4 corners of the court")
        plt.axis("on")

        # Get user input for 4 corners
        points = plt.ginput(4, timeout=0)
        plt.close()

        # Convert points to NumPy array
        keypoints = np.array(points, np.int32).reshape((-1, 1, 2))
        
        #original_h, original_w = img_rgb.shape[:2]
        # keypoints[::2] *= original_w/224.0
        # keypoints[1::2] *= original_h/224.0

        return keypoints
    
    # def draw_keypoints(self, image, keypoints):
    #     for i in range(0, len(keypoints), 2):
    #         x, y = int(keypoints[i]), int(keypoints[i+1])
            
    #         cv2.putText(image, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #         cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
    #     return image

    def draw_keypoints(self, image, keypoints):
    # Loop through each keypoint
        for i, point in enumerate(keypoints):
            # Extract x, y coordinates from the keypoint (which is a nested list)
            x, y = int(point[0][0]), int(point[0][1])
            
            # Add a label next to the point
            cv2.putText(image, str(i), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Draw a circle around the point
            cv2.circle(image, (x, y), 5, (0, 255, 255), -1)

            cv2.polylines(image, [keypoints], isClosed=True, color=(0, 255, 255), thickness=1)

        return image
    
    def draw_keypoints_on_video(self, video_frames, keypoints):
        output_video_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        return output_video_frames
