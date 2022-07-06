import queue

import yaml
from yaml.loader import SafeLoader
import cv2
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from skimage.morphology import medial_axis

from .PixelGraph import PixelGraph
from .CoordinateTransform import CoordinateTransform

def image_to_centerline(image_path, image_extension='.pgm', track_thresh=205):
    """
    Converts an image to a sequence of points representing the centerline

    Args:
        image_path (str): path to the image to be converted
        track_thresh (int): threshold for separating track from non-track (track should be white)
    """
    
    # Reading image
    img = cv2.imread(f'{image_path}{image_extension}', cv2.IMREAD_GRAYSCALE)
    _, thresh_img = cv2.threshold(img, track_thresh, 255, cv2.THRESH_BINARY)
    axis = medial_axis(thresh_img)
    
    # Converting to pixel graph and getting spanning tree and missing links
    graph = PixelGraph(axis)
    tree, missing_links = pixel_graph_to_spanning_tree_and_missing_links(graph)

    # Getting centerline
    centerline  = missing_links_and_tree_to_centerline(missing_links, tree)

    return centerline

def save_centerline_points_metric(centerline, map_path, map_extension):
    """
    Saves the centerline points in a csv in the metric reference frame
    """

    centerline_path = f'{map_path}.csv'
    image = cv2.imread(f'{map_path}{map_extension}', cv2.IMREAD_GRAYSCALE)
    with open(f'{map_path}.yaml') as f:
        map_info = yaml.load(f, Loader=SafeLoader)
    
    centerline_pixel_coords = np.array([CoordinateTransform.pixel_coords_from_flattened_index(point, image.shape) 
                                        for point in centerline])
    centerline_metric_coords = np.array([CoordinateTransform.image_to_metric(coord, np.array(map_info['origin'][:2]),
                                        image.shape, map_info['resolution']) for coord in centerline_pixel_coords])

    np.savetxt(centerline_path, centerline_metric_coords, delimiter=',')

def pixel_graph_to_spanning_tree_and_missing_links(graph: PixelGraph):
    # Initialization
    parents = {}
    q = queue.Queue()
    first = graph.get_nonzero_node()
    q.put(first)
    visited = set({first})
    missing_links = []

    # Breadth-first search
    while not q.empty():
        node = q.get()

        for adj_node in graph.adj(node):
            if adj_node not in visited:
                parents[adj_node] = (node, adj_node, graph.dist(node, adj_node))
                q.put(adj_node)
                visited.add(adj_node)
            else:
                missing_links.append((node, adj_node, graph.dist(node, adj_node)))

    G = nx.Graph()
    G.add_weighted_edges_from(parents.values())

    return G, missing_links

def missing_links_and_tree_to_centerline(missing_links, tree):
    # Get weights of rings closed by missing links
    path_weight_and_missing_links = []
    for missing_link in missing_links:
        weight, _ = nx.single_source_dijkstra(tree, missing_link[0], missing_link[1])
        ring_weight = weight + missing_link[2]
        path_weight_and_missing_links.append((ring_weight, missing_link))

    # Cluster based on path weights (the bisggest cluster should be the ring)
    X = np.array([path_weight for path_weight, _ in path_weight_and_missing_links])
    kmeans = KMeans(n_clusters=2).fit(X.reshape(-1,1))
    first_cluster_selection = kmeans.labels_.astype(bool)
    second_cluster_selection = ~first_cluster_selection
    selections = [first_cluster_selection, second_cluster_selection]

    max_average_selection_index = np.argmax([X[selection].mean() for selection in selections])
    selection_to_use = selections[max_average_selection_index]

    max_cluster_indexes = np.array(list(range(len(path_weight_and_missing_links))))[selection_to_use]
    trackline_weights_index = max_cluster_indexes[np.argmin(X[max_cluster_indexes])]
    _, trackline_link = path_weight_and_missing_links[trackline_weights_index]

    _, path = nx.single_source_dijkstra(tree, trackline_link[0], trackline_link[1])
    
    return path
