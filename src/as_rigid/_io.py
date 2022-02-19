import os
from typing import TextIO

import numpy as np
import re


def read_file(file: TextIO):
    def parse_line(_line):
        _line = re.findall(r"[-+]?\d*\.*\d+", _line)
        return np.array(_line, dtype=np.float32)

    vertices, faces = [], []

    for line in file:
        if line[0] == "#":
            # skip comments
            continue

        data = parse_line(line)
        if line[0] == "v":
            # append vertex
            vertices.append(data[:2])

        elif line[0] == "f":
            # append face
            faces.append(np.array(data[:3], dtype=np.uint32))

    vertices = np.array(vertices, dtype=np.float32)
    # shift faces by 1 due to convinience in wavefront obj
    faces = np.array(faces, dtype=np.uint32) - 1

    return len(vertices), len(faces), vertices, faces


def save_mesh(no_vertices, no_faces, vertices, faces, count):
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    file = open(os.path.join(__location__, 'man' + str(count) + '.obj'), 'w')
    file.write("#vertices: " + str(no_vertices) + "\n")
    file.write("#faces: " + str(no_faces) + "\n")
    for v in vertices:
        file.write("v " + str(v[0]) + " " + str(v[1]) + " 0\n")
    for f in faces:
        file.write("f " + str(int(f[0])) + " " + str(int(f[1])) + " " + str(int(f[2])) + "\n")
    file.close()
    print('Saved!')
