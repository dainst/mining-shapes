#!/bin/bash 

if type nvidia-smi > /dev/null || wmic path win32_VideoController get name | grep -q "NVIDIA"; then
 var="Docker_GPU"
else
 var="Docker_CPU"
fi

docker-compose -f ${var}/docker-compose.yml up