# Shape Mining Pipeline
## Description
The shape mining pipeline is designed to extract veselprofiles with corresponding metadata from book scans.
1. Extract figureID and pageID from book scan
2. Extract veselprofiles from book scan
3. Extract only characteristic shape of the veselprofiles
4. kNN approach to find most similar vesel shapes

## Getting Started
There are two Docker Containers. One for a GPU machine and one for a CPU machine.
The container hosts a jupyter server. Ther URL for accessing the server
will be shown in the terminal. Use docker-compose to run your preferred config (CPU or GPU). Example for CPU
```
docker-compose -f Docker/docker-compose.GPU.yml up
```
Access Docker shell. The container has to run for that. 
```
docker ps (to check COTAINER_ID)
docker exec -it CONTAINER_ID bash
```

## Development with Visual Studio Code and devcontainers
1. Install [Visual Studio Code](https://code.visualstudio.com/) with the following extensions:
- [Remote Development](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack)
- [Docker](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-docker)
2. Open [devcontainer](.devcontainer/devcontainer.json) file and choose in entry "dockerComposeFile" the GPU or CPU container.
3. Create following directories if you use a Linux OS:
- vscode_remote/extensions
- vscode_remote/bashhistory
- vscode_remote/insiders
4. Go to directory Docker/. Copy .template file in directory of required config (GPU/CPU) to .yml file. For example for CPU config
   ```
   cp docker-compose.CPU.dev.yml.template docker-compose.CPU.dev.yml
   ```
5. In .vscode directory copy settings.json.template to settings.json and adjust desired settings.
6. Mount required data directories in docker-compose.***Cofig***.yml
7. In VSCode press Shift+P and run "Remote-Containers:Rebuild and Reopen in Container" command.

## Cotainer filesystem
Source code is located at /home/Code <br>
Tensorflow objection detection API at /models/research/object_detection <br>

## Models 
Models for mining shapes can be downloaded at [Mining Pages](Mining_Pages/README.md)

## Run whole pipeline
Mount correct volumes to docker-compose file <br>
Run file mining_pages.py 
