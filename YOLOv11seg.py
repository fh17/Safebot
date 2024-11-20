#pip install inference_sdk

import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient


class MaskGenerator:
    def __init__(self):
        # Initialize the Roboflow inference client for the new YOLO model
        self.CLIENT = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key="vq8Qj0PykMfovdhRJOXe"
        )
        self.model_id = "safebot-box-instance-seg/1"
        self.mask = None
        self.response_data = None

    def run_inference(self, image_path):
        """Run inference on the given image and store the results."""
        # Run inference using Roboflow client
        self.response_data = self.CLIENT.infer(image_path, model_id=self.model_id)

    def generate_mask(self, image_path):
        """Generate a binary mask from the inference results."""
        # Load the original image where detections are to be drawn
        img = cv2.imread(image_path)

        # Create a black mask with the same dimensions as the original image (single channel)
        self.mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        # Loop through each detected object in the image
        for detection in self.response_data.get("predictions", []):
            # Draw the segmentation points (if any)
            if "points" in detection:
                # Convert points to an array of (x, y) coordinates
                points = np.array([(int(point["x"]), int(point["y"])) for point in detection["points"]], np.int32)

                # Reshape points to match fillPoly format
                points = points.reshape((-1, 1, 2))

                # Fill the segmentation area with white (255) on the mask
                cv2.fillPoly(self.mask, [points], 255)

        return self.mask  # Return the generated mask

    def process_image(self, image_path):
        """Run inference and generate a mask from the given image."""
        self.run_inference(image_path)  # Run inference
        return self.generate_mask(image_path)  # Generate and return the mask