#!/bin/bash

xhost +
docker start isaac-lab
docker exec -it isaac-lab /bin/bash