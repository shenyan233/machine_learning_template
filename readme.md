# A Machine Learning Template based on Pytorch Lightning
This project implements the neural network training and testing pipeline based on PyTorch Lightning. In addition to implementing the workflow of PyTorch Lightning, the project also realized adding tasks in the training process through task pool, k-fold cross-validation, saving the training results in .CVS, accepting random seeds for resumed training, and converting the model to .onnx and .tflite.

中文介绍: https://zhuanlan.zhihu.com/p/520694143?

# Usage
## Configure the network architecture and data set
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
The dataset needs to be adjusted to yourself dataset, and the 
image (*.png) name is the line number in the corresponding label.txt. network.py contains the network architecture to be trained.
You can change the file name of network.py to another name, but __init__.py needs to be changed synchronously.
The configuration parameters of the task flow is saved in tasks.json.

## Install dependencies
```bash
pip install -r requirements.txt
```
* cuda and torch need to be installed by itself

## Set the parameters
Set the parameters in ./network/{network_name}/config.json, which include model_name, dataset_path, stage, max_epochs, 
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
