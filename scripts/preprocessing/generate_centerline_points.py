from trajectory_predictor.utils.mapping import image_to_centerline, save_centerline_points_metric
import argparse

########################################################
#  Generates the centerline points given a map image   #
########################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generates the centerline points given a map image')
    parser.add_argument('--map_path',type=str, required=True, help='Path to the map')
    path = parser.parse_args().map_path

    centerline_path = f'{path}/centerline.csv'
    centerline = image_to_centerline(path)

    save_centerline_points_metric(centerline, path, '.pgm')
