# nv-tao-gui ( itao )
The QT version for NVIDIA TAO Toolkit, and it's the legacy version when I work in Innodisk called `iTAO`

## Pre-requisite
1. [Docker](https://max-c.notion.site/Install-Docker-9a0927c9b8aa4455b66548843246152f)
2. [NVIDIA-Container-Toolkit](https://max-c.notion.site/Install-NVIDIA-Container-Toolkit-For-Docker-7db1728db09e4378871303ae6c616401)
3. [NGC API Key](https://max-c.notion.site/Get-NVIDIA-NGC-API-Key-911f9d0a5e1147bf8ad42f3c0c8ca116)

## Build Docker Image
auto build the docker image
```bash
./itao.sh build
```

## Run iTAO.
activate the container with iTAO.
```bash
./itao.sh run
python3 demo --docker
```

## Demo
* Initailize
  
  <img src="assets\figures\itao_init.gif" width=80%>

* Training
  
  <img src="assets\figures\itao_train.gif" width=80%>

* Optimization
  
  <img src="assets\figures\itao_opt.gif" width=80%>

* Inference and Export
  
  <img src="assets\figures\itao_infer.gif" width=80%>


## Debug Mode
You can enable target option for debug. DEBUG_PAGE means the page from 1 to 4 and DEBUG_OPT is the feature in DEBUG_PAGE (e.g. train, kmeans, eval, etc.)
```bash
# Make sure you are in docker container
python3 demo --docker --debug --page <DEBUG_PAGE> --opt <DEBUG_OPT>
```

## Developer Mode
please check `README-DEV.md` ...
