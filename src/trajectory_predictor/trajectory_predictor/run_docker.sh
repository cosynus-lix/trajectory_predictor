dir=$(pwd)
parentdir="$(dirname "$dir")"
docker run -it --rm -v $parentdir/trajectory_generator:/trajectory_generator  predictor