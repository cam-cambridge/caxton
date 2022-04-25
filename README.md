# CAXTON: The Collaboration Autonomous Extrusion Network

_Accompanying code to the publication "Generalisable 3D Printing Error Detection and Correction via Multi-Head Neural Networks"_

## Setup

This repository allows you to easily train a multi-head residual attention neural network to classify the state of the four most important printing parameters: flow rate, lateral speed, Z offset, and hotend temperature from a single input image.

First create a Python 3 virtual environment and install the requirements. We use PyTorch 1.7.1, Torchvision 0.8.2, and CUDA 11.3 in this work.

```
virtualenv -p python3 env
source env/bin/activate
pip install -r requirements.txt
```

