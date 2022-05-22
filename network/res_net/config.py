config = {
    # requires|必填
    'model_name': 'res_net',
    'dataset_path': './dataset/cifar-100',
    'stage': 'fit',
    'max_epochs': 200,
    'batch_size': 128,
    'version_info': '',
    # optional|可选
    'precision': 16,
    # model parameters|模型参数
    'dim_in': 32,
    'num_classes': 100,
}
