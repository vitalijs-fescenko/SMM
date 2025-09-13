# SMM




## requirements
```bash

# install SAM
cd sam
pip install -e .
pip install matplotlib



```


## Docker instructions
Build
```bash
docker build -t smm_docker .
```

Run 
```bash
docker run --gpus device=0 --shm-size=8g --rm -it -p 8888:8888 -p 6006:6006 -u "$(id -u):$(id -g)" -v $(pwd):/home/appuser -v ~/Datasets:/datasets smm_docker
```
