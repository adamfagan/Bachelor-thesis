#!/bin/bash

jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root &
tensorboard --logdir=logs/ --host 0.0.0.0 --port 6006