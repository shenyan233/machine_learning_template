# A Machine Learning Template based on Pytorch Lightning
This project implements the neural network training and testing pipeline based on PyTorch Lightning. In addition to implementing the workflow of PyTorch Lightning, the project also realized adding tasks in the training process through task pool, k-fold cross-validation, saving the training results in .CVS, accepting random seeds for resumed training, and converting the model to .onnx and .tflite.

中文介绍: https://worthpen.top/blog?id=6566175b6aa58e39d9301954

# Usage

## Install dependencies
python version: 3.7-3.10
```bash
pip install -r requirements.txt
```
* cuda and torch need to be installed by itself. Recommendation: 
```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

## Configure the network architecture and dataset
The directory structure of the whole project is as follows:
```bash
.
├── dataset
│   └── {dataset_name}
│        ├── test
│        │   ├── image
│        │   │   └── *.png
│        │   └── label.txt
│        ├── train
│        │   ├── image
│        │   │   └── *.png
│        │   └── label.txt
│        ├── __init__.py
│        └── ...
├── network
│   └── {network_name}
│         ├── network.py
│         ├── __init__.py
│         └── ...
├── tasks.json
└── ...
```
The files or folders shown above need to be pre-configured. Keep the default file or folder represented by the ellipsis. 

The dataset needs to be adjusted to yourself dataset. In the above example, the image (*.png) name is the line number in the corresponding label.txt. 

You can freely adjust the save format of the dataset here, but Dataloder and other classes need to be rewritten in './dataset/{dataset name}/__init__.py'.

'network.py' contains the network architecture to be trained. You can change the file name of network.py to another name, but init.py needs to be changed synchronously. The configuration parameters of the task flow is saved in tasks.json.


## Set the parameters
Set the parameters in ./tasks.json, which include model_name, dataset_path, stage, max_epochs, 
batch_size and so on. Parameters include optional and required parameters. See the annotation in main.py for details. 
The training parameter 'stage' is 'fit' or 'test', which are represented as training phase or testing phase, respectively.

## Start Train/Test
In terminal, execute:
```bash
python3 main.py
```

## Contact
Wechat(微信): shenyan233

Twitter: @shenyan12138
