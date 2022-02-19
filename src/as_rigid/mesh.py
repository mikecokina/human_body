import cv2
import numpy as np


def draw_mesh(vertices, edges, img):
    # scaling so that it fits in the window for OpenCV coordinate system
    vertices_scaled = np.zeros(np.shape(vertices))
    vertices_scaled[:, 0] = vertices[:, 0] * -180 + 640
    vertices_scaled[:, 1] = vertices[:, 1] * -180 + 400
    vertices_scaled = vertices_scaled.astype(int)
    for edge in edges:
        start = (vertices_scaled[int(edge[0] - 1), 0], vertices_scaled[int(edge[0] - 1), 1])
        end = (vertices_scaled[int(edge[1] - 1), 0], vertices_scaled[int(edge[1] - 1), 1])
        cv2.line(img, start, end, (0, 255, 0), 1)


def get_edges(no_faces, faces):
    edges = np.zeros([no_faces * 3, 2])
    for index, face in enumerate(faces):
        edges[index * 3, :] = [face[0], face[1]]
        edges[index * 3 + 1, :] = [face[1], face[2]]
        edges[index * 3 + 2, :] = [face[0], face[2]]
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    return np.array(edges, dtype=np.uint32)
