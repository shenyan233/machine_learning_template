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
         seed,
         dataset_path,
         gpus=None,
         tpu_cores=None,
         version_nth=None,
         path_final_save=None,
         every_n_epochs=1,
         save_top_k=1,
         k_fold=5,
         kth_fold_start=0):
    """
    框架的入口函数. 包含设置超参数, 划分数据集, 选择训练或测试等流程
    该函数的参数为训练过程中需要经常改动的参数

    :param stage: 表示处于训练阶段还是测试阶段, fit表示训练, test表示测试
    :param max_epochs:
    :param batch_size:
    :param precision: 训练精度, 正常精度为32, 半精度为16, 也可以是64. 精度代表每个参数的类型所占的位数
    :param seed:

    :param dataset_path: 数据集地址, 其目录下包含数据集文件夹, 标签文件夹, 全部数据的命名list
    :param gpus:
    :param tpu_cores:
    :param version_nth: 不论是重载训练还是测试, 固定为该folds的第一个版本的版本号
    :param path_final_save:
    :param every_n_epochs: 每n个epoch设置一个检查点
    :param save_top_k:
    :param kth_fold_start: 从第几个fold开始, 若使用重载训练, 则kth_fold_start为重载第几个fold, 第一个值为0.
                           非重载训练的情况下, 可以通过调整该值控制训练的次数;
    :param k_fold:
    """
    # 经常改动的      参数    作为main的输入参数
    # 不常改动的   非通用参数    存放在config
    # 不常改动的    通用参数     直接进行声明
    # 通用参数指的是所有网络中共有的参数, 如time_sum等

    # 处理输入数据
    precision = 32 if (gpus is None and tpu_cores is None) else precision
    # 获得通用参数
    # TODO 获得最优的batch size
    num_workers = cpu_count()
    # 获得非通用参数
    config = {'dim_in': 5,
              'dim': 10,
              'res_coef': 0.5,
              'dropout_p': 0.1,
              'n_layers': 3,
              'dataset_len': 100000}
    for kth_fold in range(kth_fold_start, k_fold):
        load_checkpoint_path = get_ckpt_path(version_nth, kth_fold)
        logger = pl_loggers.TensorBoardLogger('logs/')
        dm = DataModule(batch_size=batch_size, num_workers=num_workers, k_fold=k_fold, kth_fold=kth_fold,
                        dataset_path=dataset_path, config=config)
        if stage == 'fit':
            # SaveCheckpoint的创建需要在TrainModule之前, 以保证网络参数初始化的确定性
            save_checkpoint = SaveCheckpoint(seed=seed, max_epochs=max_epochs,
                                             path_final_save=path_final_save,
                                             every_n_epochs=every_n_epochs, verbose=True,
                                             monitor='Validation acc', save_top_k=save_top_k,
                                             mode='max')
            training_module = TrainModule(config=config)
            if kth_fold != kth_fold_start or load_checkpoint_path is None:
                print('进行初始训练')
                trainer = pl.Trainer(max_epochs=max_epochs, gpus=gpus, tpu_cores=tpu_cores,
                                     logger=logger, precision=precision, callbacks=[save_checkpoint])
                training_module.load_pretrain_parameters()
            else:
                print('进行重载训练')
                trainer = pl.Trainer(max_epochs=max_epochs, gpus=gpus, tpu_cores=tpu_cores,
                                     resume_from_checkpoint='./logs/default' + load_checkpoint_path,
                                     logger=logger, precision=precision, callbacks=[save_checkpoint])
            print('训练过程中请注意gpu利用率等情况')
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
    main('fit', max_epochs=2, batch_size=32, precision=16, seed=1234, dataset_path='./dataset', k_fold=5,
         # gpus=1,
         # version_nth=8,
         kth_fold_start=4,
         )
