from typing import List

from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import sRGBColor, LabColor
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import type_metric, distance_metric


def rgb_to_cielab(rgb):
    srgb = sRGBColor(rgb[0], rgb[1], rgb[2])
    return convert_color(srgb, LabColor)


def delta_e_distance(c1, c2):
    """
    :param c1: Tuple;  (L, A, B)
    :param c2: Tuple; (L, A, B)
    :return: float;
    """
    c1_lab, c2_lab = LabColor(*c1), LabColor(*c2)
    return delta_e_cie2000(c1_lab, c2_lab)


def clab_kmeans(data: List, initial_centers: List):
    metric = distance_metric(type_metric.USER_DEFINED, func=delta_e_distance)

    # create K-Means algorithm with specific distance metric
    # initial_centers = [[4.7, 5.9], [5.7, 6.5]]
    kmeans_instance = kmeans(data, initial_centers, metric=metric)

    # run cluster analysis and obtain results
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()
    centers = kmeans_instance.get_centers()

    return clusters, centers


def standard_kmeans(data, n):
    initial_centers = kmeans_plusplus_initializer(data, n).initialize()
    instance = kmeans(data, initial_centers)
    instance.process()

    clusters = instance.get_clusters()
    centers = instance.get_centers()

    return clusters, centers

