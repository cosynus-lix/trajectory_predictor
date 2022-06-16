dir=$(pwd)
parentdir="$(dirname "$dir")"
docker run -it --rm -v $parentdir/trajectory_generator:/trajectory_generator  predictor

# docker run -it --rm -v /Users/lix/Documents/code/trajectory_predictor/:/trajectory_predictor predictor