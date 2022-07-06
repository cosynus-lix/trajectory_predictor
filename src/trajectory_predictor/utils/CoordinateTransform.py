import numpy as np

class CoordinateTransform:
    def metric_to_image(xy, origin, output_shape, resolution):
        """
        xy: numpy array of with shape (N)
        origin: where the car starts in the metric reference frame
        output_shape: number of output pixels (for now use always a square image)
        resolution: meters per pixel
        """
        assert len(output_shape) == 2 and output_shape[0] == output_shape[1]

        converted_point = np.array([origin[0], -origin[1]]) + np.array([-xy[0], xy[1]])
        scaled = converted_point * 1 / resolution

        return (np.array([0, output_shape[0]]) - scaled).astype(int)
    
    def image_to_metric(xy, origin, output_shape, resolution):
        """
        xy: numpy array of with shape (N)
        origin: where the car starts in the metric reference frame
        output_shape: number of output pixels (for now use always a square image)
        resolution: meters per pixel
        """
        assert len(output_shape) == 2 and output_shape[0] == output_shape[1]

        scaled = np.array([0, output_shape[0]]) - np.array([xy[1], xy[0]])
        converted_point = scaled * resolution
        new_coords = converted_point-np.array([origin[0], -origin[1]])
        xy = np.array([-new_coords[0], new_coords[1]])
        return xy

    def pixel_coords_from_flattened_index(index, shape):
        """
        index: flattened index of the image
        shape: shape of the image (same format as numpy)
        """
        assert len(shape) == 2
        return np.unravel_index(index, shape)

