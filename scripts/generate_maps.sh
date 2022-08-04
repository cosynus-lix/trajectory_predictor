#! /bin/bash

base_path="$(git rev-parse --show-toplevel)"
num_maps=10

# Creating maps
python $base_path/scripts/track_generator/random_trackgen.py --base_path $base_path/ --num_maps $num_maps

# Adding centerline to maps
for ((i=0; i<$num_maps; i++))
do
    clear
    echo "Generating spline for map $(expr $i + 1)/$num_maps"
    python generate_centerline_spline.py --map_path $base_path/maps/map$i
done
clear
