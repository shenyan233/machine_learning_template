from save_checkpoint import SaveCheckpoint
from data_module import DataModule
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from train_model import TrainModule


def main(stage,
         num_workers,
         max_epochs,
         batch_size,
         precision,
         seed,
         dataset_path=None,
         gpus=None,
         tpu_cores=None,
         load_checkpoint_path=None,
         save_name=None,
         path_final_save=None,
         every_n_epochs=1,
         save_top_k=1,):
    """
    框架的入口函数. 包含设置超参数, 划分数据集, 选择训练或测试等流程
    该函数的参数为训练过程中需要经常改动的参数

    :param stage: 表示处于训练阶段还是测试阶段, fit表示训练, test表示测试
    :param num_workers:
    :param max_epochs:
    :param batch_size:
    :param precision: 训练精度, 正常精度为32, 半精度为16, 也可以是64. 精度代表每个参数的类型所占的位数
    :param seed:

    :param dataset_path: 数据集地址, 其目录下包含数据集, 标签, 全部数据的命名list
    :param gpus:
    :param tpu_cores:
    :param load_checkpoint_path:
    :param save_name:
    :param path_final_save:
    :param every_n_epochs:
    :param save_top_k:
    """
    # config存放确定模型后不常改动的非通用的参数, 通用参数且不经常带动的直接进行声明
    if False:
        config = {'dataset_path': dataset_path,
                  'dim_in': 2,
                  'dim': 10,
                  'res_coef': 0.5,
                  'dropout_p': 0.1,
                  'n_layers': 2,
                  'flag': True}
    else:
        config = {'dataset_path': dataset_path,
                  'dim_in': 62,
                  'dim': 32,
                  'res_coef': 0.5,
                  'dropout_p': 0.1,
                  'n_layers': 20,
                  'flag': False}
    # TODO 获得最优的batch size
    # TODO 自动获取CPU核心数并设置num workers
    precision = 32 if (gpus is None and tpu_cores is None) else precision
    dm = DataModule(batch_size=batch_size, num_workers=num_workers, config=config)
    logger = pl_loggers.TensorBoardLogger('logs/')
    if stage == 'fit':
        training_module = TrainModule(config=config)
        save_checkpoint = SaveCheckpoint(seed=seed, max_epochs=max_epochs,
                                         save_name=save_name, path_final_save=path_final_save,
                                         every_n_epochs=every_n_epochs, verbose=True,
                                         monitor='Validation acc', save_top_k=save_top_k,
                                         mode='max')
        if load_checkpoint_path is None:
            print('进行初始训练')
            trainer = pl.Trainer(max_epochs=max_epochs, gpus=gpus, tpu_cores=tpu_cores,
                                 logger=logger, precision=precision, callbacks=[save_checkpoint])
            training_module.load_pretrain_parameters()
        else:
            print('进行重载训练')
            trainer = pl.Trainer(max_epochs=max_epochs, gpus=gpus, tpu_cores=tpu_cores,
                                 resume_from_checkpoint='./logs/default' + load_checkpoint_path,
                                 logger=logger, precision=precision, callbacks=[save_checkpoint])
        trainer.fit(training_module, datamodule=dm)
    if stage == 'test':
        if load_checkpoint_path is None:
            print('未载入权重信息，不能测试')
        else:
            print('进行测试')
            training_module = TrainModule.load_from_checkpoint(
                checkpoint_path='./logs/default' + load_checkpoint_path,
                **{'config': config})
            trainer = pl.Trainer(gpus=gpus, tpu_cores=tpu_cores, logger=logger, precision=precision)
            trainer.test(training_module, datamodule=dm)
    # 在cmd中使用tensorboard --logdir logs命令可以查看结果，在Jupyter格式下需要加%前缀


if __name__ == "__main__":
    main('fit', num_workers=8, max_epochs=5, batch_size=32, precision=16, seed=1234,
         # gpus=1,
         # load_checkpoint_path='/version_5/checkpoints/epoch=149-step=7949.ckpt',
         )
