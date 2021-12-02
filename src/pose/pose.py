# Specify the paths for the 2 files
import cv2
import numpy as np

MODE = "MPI"

if MODE is "COCO":
    proto_file = "pose/coco/pose_deploy_linevec.prototxt"
    weights_file = "pose/coco/pose_iter_440000.caffemodel"
    n_points = 18
    POSE_PAIRS = [[1, 0], [1, 2], [1, 5],
                  [2, 3], [3, 4], [5, 6],
                  [6, 7], [1, 8], [8, 9],
                  [9, 10], [1, 11], [11, 12],
                  [12, 13], [0, 14], [0, 15],
                  [14, 16], [15, 17]]

elif MODE is "MPI":
    proto_file = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    weights_file = "pose/mpi/pose_iter_160000.caffemodel"
    n_points = 15
    POSE_PAIRS = [[0, 1], [1, 2], [2, 3],
                  [3, 4], [1, 5], [5, 6],
                  [6, 7], [1, 14], [14, 8],
                  [8, 9], [9, 10], [14, 11],
                  [11, 12], [12, 13]]

else:
    raise ValueError(f"Invalid mode `{MODE}`!")

image_file = 'single.jpeg'

frame = cv2.imread(image_file)
frame_copy = np.copy(frame)
frame_width = frame.shape[1]
frame_height = frame.shape[0]
threshold = 0.1

net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)
net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)

# input image dimensions for the network
in_width = 368
in_height = 368
inp_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (in_width, in_height), (0, 0, 0), swapRB=False, crop=False)

net.setInput(inp_blob)
output = net.forward()

h, w = output.shape[2], output.shape[3]

# empty list to store the detected keypoints
points = []

for i in range(n_points):
    # confidence map of corresponding body's part
    prob_map = output[0, i, :, :]

    # find global maxima of the prob_map
    # min_value, max_value, min_location, max_location
    _, prob, _, point = cv2.minMaxLoc(prob_map)

    # scale the point to fit on the original image
    x = (frame_width * point[0]) / w
    y = (frame_height * point[1]) / h

    if prob > threshold:
        cv2.circle(frame_copy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(frame_copy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    lineType=cv2.LINE_AA)

        # add the point to the list if the probability is greater than the threshold
        points.append((int(x), int(y)))
    else:
        points.append(None)

# draw skeleton
drawned = []
for pair in POSE_PAIRS:
    part_a = pair[0]
    part_b = pair[1]

    if points[part_a] and points[part_b]:
        cv2.line(frame, points[part_a], points[part_b], (0, 255, 255), 2)

        for draw_part in [part_a, part_b]:
            if draw_part not in drawned:
                cv2.circle(frame, points[draw_part], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)


cv2.imshow('Output-Keypoints', frame_copy)
cv2.imshow('Output-Skeleton', frame)

cv2.imwrite('Output-Keypoints.jpg', frame_copy)
cv2.imwrite('Output-Skeleton.jpg', frame)

cv2.waitKey(0)

