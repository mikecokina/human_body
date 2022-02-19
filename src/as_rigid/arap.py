def find_ijlr_vertices(edge, faces):
    """
    The rotation matrix Tk is given as a transformation that maps the vertices
    around the edge to new positions as closely as possible in a least-squares
    sense. WE SAMPLE FOUR VERTICES around the edge as a context to derive the
    local transformation Tk. It is possible to sample an arbitrary number of
    vertices greater than three here, but four is the most straightforward, and we
    have found that it produces good results. An exception applies to edges on
    the boundary. In those cases, we only USE THREE VERTICES to compute Tk.

    Source: Igarashi et al., 2009

    l x--------x j
            // edge
        i x-------x r

    Find indices of vertex l and r. When edge i-j is situated at the edge of graph,
    than use only vertex l and r will be set as np.nan.

    :param edge: np.ndarray;
    :param faces: np.ndarray;
    :return: np.ndarray;
    """

    lr_indices = [np.nan, np.nan]
    count = 0
    for i, face in enumerate(faces):
        if np.any(face == edge[0]):
            if np.any(face == edge[1]):
                neighbour_index = np.where(face[np.where(face != edge[0])] != edge[1])[0][0]
                n = face[np.where(face != edge[0])]
                lr_indices[count] = int(n[neighbour_index])
                count += 1

                if count == 2:
                    break
    l_index, r_index = lr_indices
    return [l_index, r_index]


class StepOne(object):
    @staticmethod
    def compute_g_matrix(vertices, edges, faces):
        """
        The paper requires to compute expression (G.T)^{-1} @ G.T = X.
        The problem might be solved by solving equation (G.T @ G) @ X = G.T, hence
        we can simply use np.linalg.lstsq(G.T @ G, G.T, ...).

        :param vertices: np.ndarray;
        :param edges: np.ndarray;
        :param faces: np.ndarray;
        :return: Tuple[np.ndarray, np.ndarray];

        ::

            g_indices represents indices of edges that contains result
            for expression (G.T)^{-1} @ G.T in g_product
        """
        g_product = np.zeros((np.size(edges, 0), 2, 8))
        g_indices = np.zeros((np.size(edges, 0), 4))

        # Compute G_k matrix for each `k`.
        for k, edge in enumerate(edges):
            if edge.dtype not in [np.uint32, int, np.uint64]:
                raise ValueError('Invalid dtype of edge indices. Requires np.uint32, np.uint64 or int.')
            i_vert, j_vert = vertices[edge].copy()
            i_index, j_index = edge

            l_index, r_index = find_ijlr_vertices(edge, faces)
            l_vert = vertices[l_index].copy()

            if np.isnan(r_index):
                g = np.array([[i_vert[0], i_vert[1], 1, 0],
                              [i_vert[1], -i_vert[0], 0, 1],
                              [j_vert[0], j_vert[1], 1, 0],
                              [j_vert[1], -j_vert[0], 0, 1],
                              [l_vert[0], l_vert[1], 1, 0],
                              [l_vert[1], -l_vert[0], 0, 1]])

                g_indices[k, :] = [i_index, j_index, l_index, np.nan]
                x_matrix_pad = np.linalg.lstsq(g.T @ g, g.T, rcond=None)[0]
                g_product[k, :, 0:6] = x_matrix_pad[0:2, :]

            else:
                r_vert = vertices[r_index].copy()
                g = np.array([[i_vert[0], i_vert[1], 1, 0],
                              [i_vert[1], -i_vert[0], 0, 1],
                              [j_vert[0], j_vert[1], 1, 0],
                              [j_vert[1], -j_vert[0], 0, 1],
                              [l_vert[0], l_vert[1], 1, 0],
                              [l_vert[1], -l_vert[0], 0, 1],
                              [r_vert[0], r_vert[1], 1, 0],
                              [r_vert[1], -r_vert[0], 0, 1]])
                # G[k,:,:]
                g_indices[k, :] = [i_index, j_index, l_index, r_index]
                x_matrix_pad = np.linalg.lstsq(g.T @ g, g.T, rcond=None)[0]
                g_product[k, :, :] = x_matrix_pad[0:2, :]

        return g_indices, g_product


class StepTwo(object):
    pass


if __name__ == '__main__':
    import os
    import numpy as np
    from as_rigid._io import read_file
    from as_rigid.mesh import get_edges
    from matplotlib import pyplot as plt

    _location = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    yofile_object = open(os.path.join(_location, 'man.obj'), 'r')
    _nr, _nf, _r, _f = read_file(yofile_object)
    _edges = get_edges(_nf, _f)
    _edge = _edges[161]
    StepOne.compute_g_matrix(_r, _edges, _f)

    # _l_, _r_ = find_ijlr_vertices(_edge, _f, _r)
    #
    # fig, ax = plt.subplots()
    # plt.tight_layout(pad=0)
    # ax.triplot(_r.T[0], _r.T[1], _f)
    # ax.set_aspect('equal')
    # ax.axis('off')
    #
    # ax.scatter(_r[_edge].T[0], _r[_edge].T[1], c="r", s=20)
    # ax.scatter(_l_.T[0], _l_.T[1], c="g", s=20)
    # if not np.all(np.isnan(_r_)):
    #     ax.scatter(_r_.T[0], _r_.T[1], c="k", s=20)
    # plt.show()
