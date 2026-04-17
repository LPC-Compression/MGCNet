#!/bin/bash
conda env create -f environment.yaml
conda activate mgcnet
python -m pip install -r requirements.txt
