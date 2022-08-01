#! /bin/bash

# Builds dockerfile to run the predictor

cd ..
docker build -t predictor -f docker/Dockerfile .
