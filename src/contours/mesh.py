import triangle as tri
import numpy as np
from matplotlib import pyplot as plt


def triplot_2d(r, f):
    fig, ax = plt.subplots()
    plt.tight_layout(pad=0)
    ax.triplot(r.T[0], r.T[1], f)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.show()


def main():
    w, h = 1024, 960
    delta = 50
    nx, ny = w // delta + 1, h // delta + 1
    x = np.linspace(0, w, nx)
    y = np.linspace(0, h, ny)[1:]

    xs = np.concatenate([x, x[1:], np.zeros(len(y)), np.zeros(len(y) - 1) + w])
    ys = np.concatenate([np.zeros(len(x)), np.zeros(len(x) - 1) + h, y, y[:-1]])

    frame = np.array([xs, ys]).astype(int).T
    data = {"vertices": frame}
    area = np.power(delta, 2) // 2
    t = tri.triangulate(data, f'a{area}q30')

    triplot_2d(t['vertices'], t['triangles'])

    print()
    # plt.scatter(xs, ys)
    #
    # # plt.scatter(x, np.zeros(len(x)))
    # # plt.scatter(x[1:], np.zeros(len(x) - 1) + h)
    # #
    # # plt.scatter(np.zeros(len(y)), y)
    # # plt.scatter(np.zeros(len(y) - 1) + w, y[:-1])
    #
    # plt.gca().invert_yaxis()
    # plt.gca().set_aspect('equal')
    # plt.show()


if __name__ == '__main__':
    main()
