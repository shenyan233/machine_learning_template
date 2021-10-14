#!/bin/bash

module load anaconda/2020.11
module load cuda/10.2
module load cudnn/8.1.1.33_CUDA10.2

#conda create --name py37 python=3.7
source activate py37

cd run
cd machineLearningScaffold

#pip install -r requirements.txt

python main.py

#sbatch --gpus=1 ./bscc-run.sh
#当前作业ID: 67417
#查询作业: parajobs    取消作业: scancel ID