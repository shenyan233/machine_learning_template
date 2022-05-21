import importlib
import torch
from pytorch_lightning.strategies import DDPStrategy
from save_checkpoint import SaveCheckpoint
from data_module import DataModule
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from train_model import TrainModule
from multiprocessing import cpu_count
from utils import get_ckpt_path


def main(stage,
         max_epochs,
         batch_size,
         precision,
         dataset_path,
         k_fold,
         kth_fold_start,
         model_name,
         seed=None,
         gpus=None,
         tpu_cores=None,
         version_nth=None,
         path_final_save=None,
         every_n_epochs=1,
         save_top_k=1,
         version_info='None',
         accumulate_grad_batches=1,
         profiler=None,
         ):
    """
    English annotation：
    Train the entry function of the template. Includes setting training frame parameters and selecting training or
    test flow

    :param accumulate_grad_batches:
    :param stage: Indicates that you are currently in training or testing, ’fit‘ indicates training,
                  and ’test‘ indicates testing.
    :param max_epochs:
    :param batch_size:
    :param precision: Training accuracy, normal accuracy is 32, half accuracy is 16. Precision represents the number
                      of bits of each parameter's type
    :param seed:
    :param dataset_path: Data set address, whose directory contains: dataset folder, label folder, all data named list
    :param gpus:
    :param tpu_cores:
    :param version_nth: This value is the number of versions of resumed training or the number of versions
                        at which the test began
    :param path_final_save: After each update of the CKPT file, store module in a different location
    :param every_n_epochs: Set a checkpoint for every n epochs
    :param save_top_k:
    :param kth_fold_start: Start with the number of folds. If resumed training is used, kth_fold_start is
                           the number of resumed folds, with the first value being 0. In the case of
                           resumed training, the number of training can be controlled by adjusting the value.
    :param k_fold:
    :param model_name: The model name used to automatically read config
                       at the read address ./network/{model_name}/config.py
    :param profiler:


    中文注释：
    训练模板的入口函数. 包含设置训练框架参数, 选择训练或测试流程

    :param accumulate_grad_batches:
    :param stage: 表示处于训练阶段还是测试阶段, fit表示训练, test表示测试
    :param max_epochs:
    :param batch_size:
    :param precision: 训练精度, 正常精度为32, 半精度为16. 精度代表每个参数的类型所占的位数
    :param seed:
    :param dataset_path: 数据集地址, 其目录下包含数据集文件夹, 标签文件夹, 全部数据的命名list
    :param gpus:
    :param tpu_cores:
    :param version_nth: 该值为重载训练的版本数或测试开始的版本数
    :param path_final_save: 每次更新ckpt文件后, 将其存放到另一个位置
    :param every_n_epochs: 每n个epoch设置一个检查点
    :param save_top_k:
    :param kth_fold_start: 从第几个fold开始. 若使用重载训练, 则kth_fold_start为重载第几个fold, 第一个值为0.
                           非重载训练的情况下, 可以通过调整该值控制训练的次数;
    :param k_fold:
    :param model_name: 模型名称，用于自动读取config，读取地址为./network/{model_name}/config.py
    :param profiler:
    """
    # Processing input data
    # 处理输入数据
    precision = 32 if ((gpus is None or gpus == 0) and tpu_cores is None) else precision
    version_info = version_info.replace(' ', '|')
    if torch.cuda.is_available() and gpus is None and tpu_cores is None:
        gpus = 1
    # Automatic selection parameter
    # 自动选择参数
    num_workers = min([cpu_count(), 8])
    # Obtain model parameters
    # 获得模型参数
    params = importlib.import_module(f'network.{model_name}.config')
    config = params.config
    config['batch_size'] = batch_size
    for kth_fold in range(kth_fold_start, k_fold):
        print(f'amount of fold is {kth_fold}|fold的数量为{kth_fold}')
        if version_nth is not None:
            load_checkpoint_path = get_ckpt_path(version_nth)
            version_nth += 1
        else:
            load_checkpoint_path = None
        logger = pl_loggers.TensorBoardLogger('logs/')
        dm = DataModule(num_workers=num_workers, k_fold=k_fold, kth_fold=kth_fold,
                        dataset_path=dataset_path, config=config)
        # SaveCheckpoint should be created before TrainModule to ensure the deterministic initialization of network
        # parameters.
        # SaveCheckpoint的创建需要在TrainModule之前, 以保证网络参数初始化的确定性
        save_checkpoint = SaveCheckpoint(seed=seed, max_epochs=max_epochs,
                                         path_final_save=path_final_save,
                                         every_n_epochs=every_n_epochs, verbose=True,
                                         monitor='Validation acc', save_top_k=save_top_k,
                                         mode='max', version_info=version_info, config=config)
        if stage == 'fit':
            training_module = TrainModule(config=config)
            trainer = pl.Trainer(logger=logger, precision=precision, callbacks=[save_checkpoint],
                                 gpus=gpus, tpu_cores=tpu_cores, auto_select_gpus=True,
                                 # If the strategy is None, ddP_spawn strategy is used for multiple Gpus. If
                                 # a strategy is specified, the strategy is applied regardless of the number of Gpus.
                                 # 如果策略为None, 则在单GPU时无分布式, 多GPU时采用ddp_spawn策略; 如果指定了策略, 则不论GPU数量,
                                 # 均采用指定策略
                                 strategy=DDPStrategy(find_unused_parameters=False),
                                 max_epochs=max_epochs, log_every_n_steps=1,
                                 accumulate_grad_batches=accumulate_grad_batches,
                                 profiler=profiler,
                                 )
            if kth_fold != kth_fold_start or load_checkpoint_path is None:
                print('initial training|初始训练')
                training_module.load_pretrain_parameters()
                trainer.fit(training_module, datamodule=dm)
            else:
                print('resumed training|重载训练')
                trainer.fit(training_module, datamodule=dm, ckpt_path=load_checkpoint_path)
        if stage == 'test':
            if load_checkpoint_path is None:
                print('Cannot test without loading weight information|未载入权重信息，不能测试')
            else:
                print('testing|测试')
                training_module = TrainModule.load_from_checkpoint(
                    checkpoint_path=load_checkpoint_path,
                    **{'config': config})
                trainer = pl.Trainer(logger=logger, precision=precision, callbacks=[save_checkpoint],
                                     profiler=profiler,
                                     gpus=gpus, tpu_cores=tpu_cores, auto_select_gpus=True,
                                     )
                trainer.test(training_module, datamodule=dm)
        # The result can be viewed using ’tensorboard --logdir logs‘ in CMD, with the % prefix required
        # in Jupyter format
        # 在cmd中使用tensorboard --logdir logs命令可以查看结果，在Jupyter格式下需要加%前缀


if __name__ == "__main__":
    main('fit', max_epochs=200, precision=16, dataset_path='./dataset/cifar-100', model_name='res_net',
         gpus=1,
         batch_size=128, accumulate_grad_batches=1,
         k_fold=1, kth_fold_start=0,  # version_nth=3,
         version_info='baseline',
         )
