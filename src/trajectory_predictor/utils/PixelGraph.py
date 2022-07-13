
import numpy as np

class PixelGraph:
    def __init__(self, image, radius=2):
        binary_mask = image > 0
        self.nodes = np.ravel_multi_index(np.where(binary_mask), binary_mask.shape)
        self.binary_mask = binary_mask
        self.shape = binary_mask.shape
        self.radius = radius

    def adj(self, pixel_index):
        if pixel_index not in self.nodes:
            return np.array([])
        two_d_index = np.unravel_index(pixel_index, self.shape)
        lower_i = max(two_d_index[0] - self.radius, 0)
        upper_i = min(two_d_index[0] + self.radius, self.shape[0])
        lower_j = max(two_d_index[1] - self.radius, 0)
        upper_j = min(two_d_index[1] + self.radius, self.shape[1])

        selection = self.binary_mask[lower_i:upper_i, lower_j:upper_j]
        indexes = np.where(selection)
        adj = np.ravel_multi_index((lower_i + indexes[0], lower_j + indexes[1]), self.shape)
        adj = adj[adj != pixel_index] # remove the center pixel

        return adj

    def dist(self, pixel_index1, pixel_index2):
        if pixel_index1 not in self.nodes or pixel_index2 not in self.nodes:
            return np.inf
        # Unravel indexes
        i1, j1 = np.unravel_index(pixel_index1, self.shape)
        i2, j2 = np.unravel_index(pixel_index2, self.shape)

        # Euclidean distance
        return np.sqrt((i1 - i2)**2 + (j1 - j2)**2)

    def get_nonzero_node(self):
        return self.nodes[0]
    
    def get_nodes(self):
        return self.nodes.copy()
