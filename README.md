# ML-Toy-Example
This repository is to test basic ML workflows/operations in new cluster environments. It has no 'real' content.

## Install
To install the conda environment run:

```shell
# INITIATE CONDA
conda create -n TestEnv python=3.8
conda activate TestEnv

# INSTALL SOME REQUIREMENTS
pip install numpy 
pip install scipy 
pip install pandas 
pip install opencv-python 
pip install sklearn
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts

# INSTALL ML REQUIREMENTS
pip install torch
pip install torchvision
pip install pytorch_lightning

# DEPENDING ON YOUR TORCH AND CUDA VERSION
python -c "import torch; print('TORCH VERSION:', torch.__version__)"
EITHER: pip install torch-scatter  -f https://data.pyg.org/whl/torch-1.10.0+cu102.html
OR: pip install torch-scatter  -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
```
