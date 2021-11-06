#!/bin/bash

#cd run
#cd machineLearningTemplate

module load anaconda/2020.11
module load cuda/11.1
module load cudnn/8.1.1.33_CUDA11.1

#conda create --name py39 python=3.9
source activate py39

#pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
#pip install -r requirements.txt

python main.py

#sbatch --gpus=1 ./bscc-run.sh
#当前作业ID: 67417
#查询作业: parajobs    取消作业: scancel ID