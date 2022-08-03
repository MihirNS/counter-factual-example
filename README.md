## Setup

### After cloing repository, run following command in workspace

```bash
docker build -t carla-recourse .
```

### This command will create docker image with the data set and algorithm script.

## Run algorithm

```bash
docker run --rm -i carla-recourse bash <<< "python3 -W ignore algo_llps.py"
```