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

def filter_track_on_image(image_path, image_extension='.pgm'):
    """
    Converts an image to a sequence of points representing the centerline

    Args:
        image_path (str): path to the image to be converted
        image_extension (str): extension of the image
        track_thresh (int): threshold for separating track from non-track (track should be white)
    """

    # Reading image
    img = cv2.imread(f'{image_path}{image_extension}', cv2.IMREAD_GRAYSCALE)

    # Converting to pixel graph and getting spanning tree and missing links
    graph = PixelGraph(img == 0, radius=3)
    
    # Getting all bigger loops in graph
    loops = find_big_loops(graph)
    filled_loops = [path_to_filled_image(loop, img.shape) for loop in loops]
    filled_loops = simplify_images(filled_loops)

    if len(filled_loops) == 2:
        filled_loops = sorted(filled_loops, key=lambda x: np.sum(x != 0))
        return filled_loops[1]-filled_loops[0]
    else:
        raise Exception('Failed to find the two loops defining the track')
    
def simplify_images(images):
    """
    Simplifies a list of images merging similar images
    """

    reduced_images_list = []
    for image in images:
        found_close_image = False
        for i in range(len(reduced_images_list)):
            image_set_item = reduced_images_list[i]
            if np.sum(image != image_set_item) < 1000:
                found_close_image = True
                image[image != image_set_item] = 0
                reduced_images_list[i] = image
        if not found_close_image:
            reduced_images_list.append(image)
    return reduced_images_list

def path_to_filled_image(loop, shape):
    img = np.zeros(shape)
    two_d_indexes = np.unravel_index(loop, shape)
    arr = np.array(two_d_indexes).T
    arr[:, 0], arr[:, 1] = arr[:, 1].copy(), arr[:, 0].copy()
    cv2.fillPoly(img, pts=[arr], color=255)
    return img

def find_big_loops(graph, thresh=100):
    """
    Finds spanning trees in a graph
    """
    nodes = graph.get_nodes()
    visited_nodes = set()

    loops = []

    for node in nodes:
        if node in visited_nodes:
            continue
        pg, links = pixel_graph_to_spanning_tree_and_missing_links(graph, node)
        if len(pg.nodes) < thresh:
            continue
        [visited_nodes.add(node) for node in pg.nodes]

        path = missing_links_and_tree_to_bigger_loop(links, pg)
        loops.append(path)
    return loops

def image_to_centerline(image_path, image_extension='.pgm', track_thresh=205):
    """
    Converts an image to a sequence of points representing the centerline

    Args:
        image_path (str): path to the image to be converted
        image_extension (str): extension of the image
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
    centerline  = missing_links_and_tree_to_bigger_loop(missing_links, tree)

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

def pixel_graph_to_spanning_tree_and_missing_links(graph: PixelGraph, first=None):
    # Initialization
    parents = {}
    q = queue.Queue()
    if first is None:
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

def missing_links_and_tree_to_bigger_loop(missing_links, tree):
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
    trackline_weights_index = max_cluster_indexes[np.argmax(X[max_cluster_indexes])]
    _, trackline_link = path_weight_and_missing_links[trackline_weights_index]

    _, path = nx.single_source_dijkstra(tree, trackline_link[0], trackline_link[1])
    
    return path
