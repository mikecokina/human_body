import json
import math
from collections import defaultdict

import cv2
import os
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_landmark = mp_pose.PoseLandmark

points_of_interest = [
    mp_landmark.MOUTH_RIGHT,
    mp_landmark.MOUTH_LEFT,
    mp_landmark.RIGHT_SHOULDER,
    mp_landmark.LEFT_SHOULDER
]


def get_pose(image):
    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5
    ) as pose:
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        segmentation_mask = results.segmentation_mask
        pose_landmarks = results.pose_landmarks
        pose_world_landmarks = results.pose_world_landmarks
        return segmentation_mask, pose_landmarks, pose_world_landmarks


def read_json(path):
    with open(path, "r") as f:
        content = f.read()
        return json.loads(content)


def angle(line_1: np.array, line_2: np.array):
    from pypex.poly2d import line
    line_1 = line.Line(line_1)
    line_2 = line.Line(line_2)
    return line_1.angle(line_2, degrees=True)


def check_std_pose_angles(pose_landmarks):
    landmarks = pose_landmarks.landmark
    # visible all points of interest:
    for poi in points_of_interest:
        if landmarks[poi].visibility < 0.8:
            return False
    x_axis = np.array([[0.0, 0.0], [1.0, 0.0]])

    # angles of interest
    # mouth line and x axis angle
    mouth_line = np.array(
        [[landmarks[mp_landmark.MOUTH_RIGHT].x, landmarks[mp_landmark.MOUTH_RIGHT].y],
         [landmarks[mp_landmark.MOUTH_LEFT].x, landmarks[mp_landmark.MOUTH_LEFT].y]]
    )
    if angle(mouth_line, x_axis) > 5:
        return False

    # shoulders line and x axis angle
    shoulders_line = np.array(
        [[landmarks[mp_landmark.RIGHT_SHOULDER].x, landmarks[mp_landmark.RIGHT_SHOULDER].y],
         [landmarks[mp_landmark.LEFT_SHOULDER].x, landmarks[mp_landmark.LEFT_SHOULDER].y]]
    )
    if angle(shoulders_line, x_axis) > 5:
        return False

    # shoulder to elbow and elbow to wrist angle
    shoulder_elbow_line_left = np.array(
        [[landmarks[mp_landmark.LEFT_SHOULDER].x, landmarks[mp_landmark.LEFT_SHOULDER].y],
         [landmarks[mp_landmark.LEFT_ELBOW].x, landmarks[mp_landmark.LEFT_ELBOW].y]]
    )

    shoulder_elbow_line_right = np.array(
        [[landmarks[mp_landmark.RIGHT_SHOULDER].x, landmarks[mp_landmark.RIGHT_SHOULDER].y],
         [landmarks[mp_landmark.RIGHT_ELBOW].x, landmarks[mp_landmark.RIGHT_ELBOW].y]]
    )

    elbow_wrist_line_right = np.array(
        [[landmarks[mp_landmark.RIGHT_ELBOW].x, landmarks[mp_landmark.RIGHT_ELBOW].y],
         [landmarks[mp_landmark.RIGHT_WRIST].x, landmarks[mp_landmark.RIGHT_WRIST].y]]
    )

    elbow_wrist_line_left = np.array(
        [[landmarks[mp_landmark.LEFT_ELBOW].x, landmarks[mp_landmark.LEFT_ELBOW].y],
         [landmarks[mp_landmark.LEFT_WRIST].x, landmarks[mp_landmark.LEFT_WRIST].y]]
    )

    right_angle = np.degrees(np.pi) % angle(shoulder_elbow_line_right, elbow_wrist_line_right)
    left_angle = np.degrees(np.pi) % angle(shoulder_elbow_line_left, elbow_wrist_line_left)

    if right_angle > 5 or left_angle > 5:
        return False

    # elbow and hip to x axis angle
    elbow_hip_line_left = np.array(
        [[landmarks[mp_landmark.LEFT_ELBOW].x, landmarks[mp_landmark.LEFT_ELBOW].y],
         [landmarks[mp_landmark.LEFT_HIP].x, landmarks[mp_landmark.LEFT_HIP].y]]
    )
    elbow_hip_line_right = np.array(
        [[landmarks[mp_landmark.RIGHT_ELBOW].x, landmarks[mp_landmark.RIGHT_ELBOW].y],
         [landmarks[mp_landmark.RIGHT_HIP].x, landmarks[mp_landmark.RIGHT_HIP].y]]
    )

    right_angle = angle(elbow_hip_line_right, x_axis)
    left_angle = 180 - angle(elbow_hip_line_left, x_axis)

    if np.abs(right_angle - left_angle) > 5:
        return False

    return True


def lookup():
    _dirname = os.path.dirname(__file__)
    image_file = os.path.join(_dirname, "reference.jpg")
    frame = cv2.imread(image_file)

    cap = cv2.VideoCapture(0)

    try:
        with mp_pose.Pose(
                min_detection_confidence=0.8,
                min_tracking_confidence=0.5
        ) as pose:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    continue

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                frame.flags.writeable = False
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame)
                pose_landmarks = results.pose_landmarks

                if pose_landmarks is not None:
                    _is_match = check_std_pose_angles(pose_landmarks)
                    if _is_match:
                        frame.flags.writeable = True
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                        image_width, image_height = frame.shape[1], frame.shape[0]
                        for idx, data in enumerate(pose_landmarks.landmark):
                            x_px = min(math.floor(data.x * image_width), image_width - 1)
                            y_px = min(math.floor(data.y * image_height), image_height - 1)
                            cv2.circle(frame, (int(x_px), int(y_px)), 3, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)
                            cv2.imwrite("output.png", frame)
                        break

                # Draw the pose annotation on the image.
                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    frame,
                    pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                # Flip the image horizontally for a selfie-view display.
                cv2.imshow('MediaPipe Pose', cv2.flip(frame, 1))
                if cv2.waitKey(5) & 0xFF == 27:
                    break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()

    # image_width, image_height = frame.shape[1], frame.shape[0]
    # _, pose_landmarks, _ = get_pose(frame)
    # _is_match = check_std_pose_angles(pose_landmarks)
    #
    # print(_is_match)
    #
    # for idx, data in enumerate(pose_landmarks.landmark):
    #     x_px = min(math.floor(data.x * image_width), image_width - 1)
    #     y_px = min(math.floor(data.y * image_height), image_height - 1)
    #     cv2.circle(frame, (int(x_px), int(y_px)), 3, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)
    # cv2.imshow("image", frame)
    # cv2.waitKey()


if __name__ == '__main__':
    lookup()
