import matplotlib.pyplot as plt
import mediapipe as mp
import cv2
import numpy as np
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList, NormalizedLandmark

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
landmarks_enum = mp_pose.PoseLandmark

BG_COLOUR = [192, 192, 192]
IMAGE_FILE = "../poi/reference.jpg"
TARGET_FILE = "../poi/target.jpg"


POIs = [landmarks_enum.NOSE, landmarks_enum.LEFT_EYE, landmarks_enum.RIGHT_EYE,
        landmarks_enum.LEFT_SHOULDER, landmarks_enum.RIGHT_SHOULDER,
        landmarks_enum.LEFT_KNEE, landmarks_enum.RIGHT_KNEE,
        landmarks_enum.LEFT_HIP, landmarks_enum.RIGHT_HIP,
        landmarks_enum.LEFT_ANKLE, landmarks_enum.RIGHT_ANKLE,
        landmarks_enum.MOUTH_LEFT, landmarks_enum.MOUTH_RIGHT,
        landmarks_enum.LEFT_ELBOW, landmarks_enum.RIGHT_ELBOW,
        landmarks_enum.LEFT_WRIST, landmarks_enum.RIGHT_WRIST]


def hide_not_interested(pose_landmarks):
    for landmark_index, landmark in enumerate(pose_landmarks.landmark):
        if landmark_index not in POIs:
            landmark.visibility = 0.0
    return pose_landmarks


def get_reference():
    landmarks = get_landmarks(IMAGE_FILE)
    landmarks = hide_not_interested(landmarks)
    return landmarks


def get_landmarks(im_file: str):
    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.7
    ) as pose:
        image = cv2.imread(im_file)
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return results.pose_landmarks


def pad(v):
    return np.hstack([v, np.ones((v.shape[0], 1))])


def unpad(v):
    return v[:, :-1]


def affine_transformation(reference, target):
    model_features = np.array(reference)
    input_features = np.array(target)

    # In order to solve the augmented matrix (incl translation),
    # it's required all vectors are augmented with a "1" at the end
    # -> Pad the features with ones, so that our transformation can do translations too

    # Pad to [[ x y 1] , [x y 1]]
    y = pad(model_features)
    x = pad(input_features)

    # Solve the least squares problem X * A = Y
    # and find the affine transformation matrix A.
    a_matrix, res, rank, s = np.linalg.lstsq(x, y, rcond=None)
    a_matrix[np.abs(a_matrix) < 1e-10] = 0  # set really small values to zero

    return a_matrix


def landmarks_to_vector(landmarks):
    result = []
    for poi in POIs:
        x = landmarks.landmark[poi].x
        y = landmarks.landmark[poi].y
        result.append([x, y])
    return np.array(result)


def does_match_visibility(reference, target, threshold=0.8):
    for r, t in zip(reference.landmark, target.landmark):
        if r.visibility < threshold and t.visibility < threshold:
            pass
        elif not (r.visibility > threshold and t.visibility > threshold):
            return False
    return True


def transform(matrix, x):
    return unpad(np.dot(pad(x), matrix))


def poi_vector_to_normalized_landmarks(vector: np.array):
    lms = [NormalizedLandmark(x=0.0, y=0.0, z=0.0, visibility=0.0) for _ in landmarks_enum]
    for idx, poi in enumerate(POIs):
        x, y = vector[idx]
        lms[poi].x, lms[poi].y, lms[poi].visibility = x, y, 1.0
    normalized_landmark_list = NormalizedLandmarkList(landmark=lms)
    return normalized_landmark_list


def plot_transofrmations(vector_reference, transformed_target, vector_target=None):
    xs_r, ys_r = vector_reference.T[0], vector_reference.T[1]
    if vector_target is not None:
        xs_t, ys_t = vector_target.T[0], vector_target.T[1]
        plt.scatter(xs_t, ys_t, c="b")

    xs_tr, ys_tr = transformed_target.T[0], transformed_target.T[1]

    plt.scatter(xs_r, ys_r, c="r")
    plt.scatter(xs_tr, ys_tr, c="g")
    ax = plt.gca()
    ax.invert_yaxis()
    plt.show()


def rescale(vector: np.array):
    return (vector - np.min(vector)) / (np.max(vector) - np.min(vector))


def is_close(reference, target, threshold=0.05):
    distances = [np.linalg.norm(r - t) for r, t in zip(reference, target)]
    condition = np.less(distances, threshold)
    return np.all(condition)


def is_match(frame, reference):
    is_visible = does_match_visibility(reference, frame)
    if not is_visible:
        return False

    frame_v = landmarks_to_vector(frame)
    reference_v = landmarks_to_vector(reference)

    affine_matrix = affine_transformation(reference_v, frame_v)
    rotation_angle = np.degrees(np.arccos(affine_matrix.T[0][0]))

    frame_transformed = transform(affine_matrix, frame_v)

    reference_norm = rescale(reference_v)
    frame_transformed_norm = rescale(frame_transformed)

    # normalized_landmarks = poi_vector_to_normalized_landmarks(frame_transformed_norm)
    close = is_close(reference_norm, frame_transformed_norm)

    return close


def main():
    """
    affine transformation of vector x -> y
    y = f(x) = Ax + b

    [y, 1] = [A, 0 0 0 .. 0 | [b, 1] ][x, 1]
    """
    reference = get_reference()

    # target = get_landmarks(TARGET_FILE)
    # target = hide_not_interested(target)

    cap = cv2.VideoCapture(0)

    try:
        with mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
        ) as pose:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    print("Empty")
                    continue

                frame.flags.writeable = False
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(frame)
                pose_landmarks = result.pose_landmarks

                if pose_landmarks is not None:
                    pose_landmarks = hide_not_interested(pose_landmarks)
                    _is_match = is_match(pose_landmarks, reference)
                    print(_is_match)

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

            # match = is_match(target, reference)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()


















    # image = cv2.imread(TARGET_FILE)
    # annotated_image = image.copy()
    #
    # # Draw pose landmarks on the image.
    # mp_drawing.draw_landmarks(
    #     annotated_image,
    #     normalized_landmarks,
    #     mp_pose.POSE_CONNECTIONS,
    #     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # cv2.imshow('reference-pose.png', annotated_image)
    # cv2.waitKey()


if __name__ == '__main__':
    main()
