# Shape Mining Pipeline
## Description
The shape mining pipeline is designed to extract veselprofiles with corresponding metadata from book scans.
1. Extract figureID and pageID from book scan
2. Extract veselprofiles from book scan
3. Extract only characteristic shape of the veselprofiles
4. kNN approach to find most similar vesel shapes

## Getting Started
There are two Docker Containers. One for a GPU machine and one for a CPU machine.
Install and run docker container. Container hosts a jupyter server. Ther URL for accessing the server
will be shown in the terminal.
```
chmod +x start_docker.sh
./start_docker.sh
```
or use docker-compose for your preferred config (CPU or GPU). Example for CPU
```
docker-compose -f Docker_CPU/docker-compose.yml up
```
Access Docker shell. The container has to run for that. 
```
docker ps (to check COTAINER_ID)
docker exec -it CONTAINER_ID bash
```
Tip: Use VSCode with Remote Explorer extension for development.
## Cotainer filesystem
Source code is located at /home/Code <br>
Tensorflow objection detection API at /models/research/object_detection <br>

## Models 
Models for mining shapes can be downloaded at [Mining Pages](Mining_Pages/README.md)

## Run whole pipeline
Mount correct volumes to docker-compose file <br>
Run file mining_pages.py 
