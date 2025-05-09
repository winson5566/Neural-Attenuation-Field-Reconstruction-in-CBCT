# Architectural Enhancements for Neural Attenuation Field Reconstruction in CBCT
(COSC440 Deep-Learning Project Report)

## Setup

We recommend using [Conda](https://docs.conda.io/en/latest/miniconda.html) to set up an environment.

``` sh
# Create environment

conda create -n cosc440_project python=3.10 -y
conda activate cosc440_project

# Install tensorflow
pip install tensorflow==2.15.0

# Install other packages
pip install -r requirements.txt
```

## Training and evaluation

Download four CT datasets from [here](https://drive.google.com/drive/folders/1BJYR4a4iHpfFFOAdbEe5O_7Itt1nukJd?usp=sharing). Put them into the `./data/ct_data` folder.

Experiments settings are stored in `./config` folder.

For example, train NAF with `chest_50` dataset:

``` sh
python main.py --config ./config/chest_50.yaml
```
or
``` sh
nohup python main.py --config ./config/chest_50.yaml > train.log 2>&1 &
```
*Note: It may take minutes to compile the hash encoder module for the first time.*

The evaluation outputs will be saved in `./data/out` folder.
