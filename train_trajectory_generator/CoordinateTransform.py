import numpy as np

class CoordinateTransform:
    def metric_to_image(xy, origin, output_shape, resolution):
        """
        xy: numpy array of with shape (N)
        origin: where the car starts
        output_shape: number of output pixels (for now use always a square image)
        resolution: meters per pixel
        """
        assert len(output_shape) == 2 and output_shape[0] == output_shape[1]

        converted_point = np.array([origin[0], -origin[1]]) + np.array([-xy[0], xy[1]])
        scaled = converted_point * 1 / resolution

        return (np.array([0, output_shape[0]]) - scaled).astype(int)
