# A Machine Learning Template based on Pytorch Lightning
This project implements the neural network training and testing pipeline based on PyTorch Lightning. In addition to implementing the workflow of PyTorch Lightning, the project also realized adding tasks in the training process through task pool, k-fold cross-validation, saving the training results in .CVS, accepting random seeds for resumed training, and converting the model to .onnx and .tflite.

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
│        └── ...
├── network
│   └── {network_name}
│         ├── network.py
│         └── ...
└── ...
```
The files or folders represented in the above target need to be pre-configured. The dataset needs to be adjusted to yourself dataset, and the image (*.png) name is the line number in the corresponding label.txt. network.py contains the network architecture to be trained.

## Configure the environment
```bash
pip install -r requirements.txt
```
* cuda and torch need to be installed by itself

## Set the parameters
Set the parameters in ./network/{network_name}/config.json, which include model_name, dataset_path, stage, max_epochs, batch_size and so on。Parameters include optional and required parameters. The training parameter 'stage' is 'fit' or 'test', which are represented as training phase or testing phase, respectively.

## Start Train/Test
In terminal, execute:
```bash
python3 main.py
```

## Contact
Wechat(微信): shenyan233
Twitter: @shenyan12138
