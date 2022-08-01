#! /bin/bash

docker run --rm -v $(git rev-parse --show-toplevel):/trajectory_predictor -ti predictor
