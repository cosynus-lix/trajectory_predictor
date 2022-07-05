
import numpy as np
from skimage.graph import pixel_graph

class PixelGraph:
    def __init__(self, image):
        binary_mask = image > 0
        self.graph, self.nodes = pixel_graph(binary_mask, mask=None, edge_function=None, connectivity=2)

    def adj(self, pixel_index):
        if pixel_index not in self.nodes:
            return np.array([])
        index = np.where(self.nodes == pixel_index)[0]
        adj_nonzero = self.graph[index,:].nonzero()[1]
        return self.nodes[adj_nonzero]

    def dist(self, pixel_index1, pixel_index2):
        if pixel_index1 not in self.nodes or pixel_index2 not in self.nodes:
            return np.inf
        index1 = np.where(self.nodes == pixel_index1)[0]
        index2 = np.where(self.nodes == pixel_index2)[0]
        return self.graph[index1, index2].item()

    def get_nonzero_node(self):
        return self.nodes[0]
