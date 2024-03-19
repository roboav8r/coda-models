#!/bin/bash
set -e

exec "$@"

echo "CODa detector entrypoint"

source ~/.bashrc
source /opt/ros/galactic/setup.bash

cd tools
python3 ros_demo.py --pc $PC_TOPIC --ds_rate 4