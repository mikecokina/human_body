from typing import NamedTuple

import cv2
import os
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

BG_COLOR = (192, 192, 192) # gray
image_file = os.path.join(os.path.dirname(__file__), "single.jpeg")

with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5
) as pose:
    image = cv2.imread(image_file)
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    segmentation_mask = results.segmentation_mask
    pose_landmarks = results.pose_landmarks
    pose_world_landmarks = results.pose_world_landmarks

    # how to transform coordinates of given point to original image space (W x H space)
    #   results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width'
    #   results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height'

    annotated_image = image.copy()
    # Draw segmentation on the image. (show only person segmentation)
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack((segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    annotated_image = np.where(condition, annotated_image, bg_image)

    # Draw pose landmarks on the image.
    mp_drawing.draw_landmarks(
        annotated_image,
        pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    cv2.imwrite('Output-Keypoints-Mediapipe.png', annotated_image)

    # Plot pose world landmarks.
    # mp_drawing.plot_landmarks(pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

    # `pose_world_landmarks` contain 3D estimation of pose in COCO format.
    # indices are axesible in same way as standard 2D landmarks via enum mp_pose.PoseLandmark.<VALUE>

