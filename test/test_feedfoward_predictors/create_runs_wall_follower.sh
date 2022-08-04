#! /bin/bash

""" SCRIPT DEPRACATED, NEEDS TO PLACE MAP DATA 
DIRECLY IN THE ROOT OF THE MAPS DIRECTORY """

cd ../

# Creating maps
python ../scripts/track_generator/random_trackgen.py --base_path ../ --num_maps 10

# Rename maps 6, 8, 9 to 5, 6, 7
for i in "65" "86" "97"
do
    for j in png yaml pgm
    do
    mv ../maps/map${i:0:1}.$j ../maps/map${i:1:1}.$j
    done
    mv ../centerline/map${i:0:1}.csv ../centerline/map${i:1:1}.csv
done

for i in {0..7}
do
    python ../scripts/trajectory_generator_wall_follower.py --map_index $i --speed 3
done

# Cleanup
rm trajectory.png
