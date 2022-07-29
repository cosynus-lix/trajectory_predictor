FROM nvidia/cuda:10.0-base-ubuntu18.04

# Update and add repositories
RUN apt-get -y update \
    && apt-get install -y software-properties-common \
    && apt-get -y update \
    && add-apt-repository universe

# Needed for opencv
RUN apt-get install -y libgl1

# Python
RUN apt-get -y update \
    && apt-get -y install python3.8-dev \
    && apt-get -y install python3-pip

RUN python3.8 -m pip install pip --upgrade

RUN rm -f /usr/bin/python3 && ln -s /usr/bin/python3.8 /usr/bin/python3

RUN apt-get install python3.8-venv

RUN python3 -m venv /venv

COPY / /trajectory_predictor/

# To make source work
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

RUN source /venv/bin/activate && \
    pip install pip --upgrade && \
    pip install darts

RUN source /venv/bin/activate && \
    cd /trajectory_predictor && pip install -e .
