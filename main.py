import importlib
import json

import torch
from pytorch_lightning.strategies import DDPStrategy
from save_checkpoint import SaveCheckpoint
from data_module import DataModule
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from train_model import TrainModule
from multiprocessing import cpu_count
from utils import get_ckpt_path

"""
English config annotation：
    @requires
    :param model_name:
    :param dataset_path: Data set address, whose directory contains: dataset folder, label folder, all data named list
    :param stage: Indicates that you are currently in training or testing, ’fit‘ indicates training,
                  and ’test‘ indicates testing.
    :param max_epochs:
    :param batch_size:
    @optional
    :param accumulate_grad_batches:
    :param gpus:
    :param tpu_cores:
    :param precision: Training accuracy, normal accuracy is 32, half accuracy is 16. Precision represents the number
                      of bits of each parameter's type
    :param seed:
    :param k_fold:
    :param kth_fold_start: Start with the number of folds. If resumed training is used, kth_fold_start is
                           the number of resumed folds, with the first value being 0. In the case of
                           resumed training, the number of training can be controlled by adjusting the value.
    :param version_nth: This value is the number of versions of resumed training or the number of versions
                        at which the test began
    :param every_n_epochs: Set a checkpoint for every n epochs
    :param save_top_k:
    :param path_final_save: After each update of the CKPT file, store module in a different location
    :param profiler:

中文config注释：
    @必填
    :param model_name:
    :param dataset_path: 数据集地址, 其目录下包含数据集文件夹, 标签文件夹, 全部数据的命名list
    :param stage: 表示处于训练阶段还是测试阶段, fit表示训练, test表示测试
    :param max_epochs:
    :param batch_size:
    @可选
    :param version_info:
    :param accumulate_grad_batches:
    :param gpus:
    :param tpu_cores:
    :param precision: 训练精度, 正常精度为32, 半精度为16. 精度代表每个参数的类型所占的位数
    :param seed:
    :param k_fold:
    :param kth_fold_start: 从第几个fold开始. 若使用重载训练, 则kth_fold_start为重载第几个fold, 第一个值为0.
                           非重载训练的情况下, 可以通过调整该值控制训练的次数;
    :param version_nth: 该值为重载训练的版本数或测试开始的版本数
    :param every_n_epochs: 每n个epoch设置一个检查点
    :param save_top_k:
    :param path_final_save: 每次更新ckpt文件后, 将其存放到另一个位置
    :param profiler:
"""
default_config = {
    'version_info': '',
    'gpus': None,
    'tpu_cores': None,
    'accumulate_grad_batches': 1,
    'k_fold': 1,
    'kth_fold_start': 0,
    'precision': 32,
    'version_nth': None,
    'seed': None,
    'path_final_save': None,
    'every_n_epochs': 1,
    'save_top_k': 1,
    'profiler': None,
}


def main(config):
    # Setting default config
    # 配置默认config
    config = {**default_config, **config}
    # Automatic setting config
    # 自动配置参数
    if torch.cuda.is_available() and config['gpus'] is None and config['tpu_cores'] is None:
        config['gpus'] = 1
    if (config['gpus'] is None or config['gpus'] == 0) and config['tpu_cores'] is None:
        config['precision'] = 32
    for kth_fold in range(config['kth_fold_start'], config['k_fold']):
        print(f'amount of fold is {kth_fold}|fold的数量为{kth_fold}')
        if config['version_nth'] is not None:
            load_checkpoint_path = get_ckpt_path(config['version_nth'])
            config['version_nth'] += 1
        else:
            load_checkpoint_path = None
        logger = pl_loggers.TensorBoardLogger('logs/')
        dm = DataModule(num_workers=min([cpu_count(), 8]), config=config)
        # SaveCheckpoint should be created before TrainModule to ensure the deterministic initialization of network
        # parameters.
        # SaveCheckpoint的创建需要在TrainModule之前, 以保证网络参数初始化的确定性
        save_checkpoint = SaveCheckpoint(monitor='Validation acc', mode='max', config=config)
        if config['stage'] == 'fit':
            training_module = TrainModule(config=config)
            trainer = pl.Trainer(logger=logger, precision=config['precision'], callbacks=[save_checkpoint],
                                 gpus=config['gpus'], tpu_cores=config['tpu_cores'], auto_select_gpus=True,
                                 # If the strategy is None, ddP_spawn strategy is used for multiple Gpus. If
                                 # a strategy is specified, the strategy is applied regardless of the number of Gpus.
                                 # 如果策略为None, 则在单GPU时无分布式, 多GPU时采用ddp_spawn策略; 如果指定了策略, 则不论GPU数量,
                                 # 均采用指定策略
                                 strategy=DDPStrategy(find_unused_parameters=False),
                                 max_epochs=config['max_epochs'], log_every_n_steps=1,
                                 accumulate_grad_batches=config['accumulate_grad_batches'],
                                 profiler=config['profiler'],
                                 )
            if kth_fold != config['kth_fold_start'] or load_checkpoint_path is None:
                print('initial training|初始训练')
                training_module.load_pretrain_parameters()
                trainer.fit(training_module, datamodule=dm)
            else:
                print('resumed training|重载训练')
                trainer.fit(training_module, datamodule=dm, ckpt_path=load_checkpoint_path)
        if config['stage'] == 'test':
            if load_checkpoint_path is None:
                print('Cannot test without loading weight information|未载入权重信息，不能测试')
            else:
                print('testing|测试')
                training_module = TrainModule.load_from_checkpoint(
                    checkpoint_path=load_checkpoint_path,
                    **{'config': config})
                trainer = pl.Trainer(logger=logger, precision=config['precision'], callbacks=[save_checkpoint],
                                     profiler=config['profiler'],
                                     gpus=config['gpus'], tpu_cores=config['tpu_cores'], auto_select_gpus=True,
                                     )
                trainer.test(training_module, datamodule=dm)
        # The result can be viewed using ’tensorboard --logdir logs‘ in CMD, with the % prefix required
        # in Jupyter format
        # 在cmd中使用tensorboard --logdir logs命令可以查看结果，在Jupyter格式下需要加%前缀


if __name__ == "__main__":
    model_name = 'res_net'
    while True:
        # Obtain all parameters
        # 获得全部参数
        with open(f"./network/{model_name}/config.json", "r") as f:
            configs = json.load(f)
        if len(configs) == 0:
            print('over|结束')
            break
        current_key = str(min([int(i) for i in list(configs.keys())]))
        config = configs[current_key]
        main(config=config)
        with open(f"./network/{model_name}/config.json", "w") as f:
            del configs[current_key]
            f.write(json.dumps(configs, indent=2, ensure_ascii=False))
