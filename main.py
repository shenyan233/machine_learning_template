import json
import sys
from datetime import datetime
import argparse
from pytorch_lightning.accelerators import find_usable_cuda_devices
from save_checkpoint import SaveCheckpoint
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
import importlib
from data_module import DataModule
from multiprocessing import cpu_count
from utils import get_ckpt_path
import torch

"""
English config annotation：
    @requires
    :param model_name:
    :param dataset_name: Directory contains: dataset folder, label folder, all data named list
    :param stage: Indicates that you are currently in training or testing, ’fit‘ indicates training,
                  and ’test‘ indicates testing.
    :param max_epochs:
    :param batch_size:
    @optional
    :param version_info:
    :param accelerator:
    :param devices:
    :param accumulate_grad_batches:
    :param k_fold:
    :param kth_fold_start: Start with the number of folds. If resumed training is used, kth_fold_start is
                           the number of resumed folds, with the first value being 0. In the case of
                           resumed training, the number of training can be controlled by adjusting the value.
    :param precision: Training accuracy, normal accuracy is 32, half accuracy is 16. Precision represents the number
                      of bits of each parameter's type. Double precision (64, '64' or '64-true'), full precision (32,
                       '32' or '32-true'), 16bit mixed precision (16, '16', '16-mixed') or bfloat16 mixed precision 
                       ('bf16', 'bf16-mixed'). Can be used on CPU, GPU, TPUs, HPUs or IPUs.
    :param log_name: 
    :param version_nth: This value is the number of versions of resumed training or the number of versions
                        at which the test began
    :param seed:
    :param path_final_save: After each update of the CKPT file, store module in a different location
    :param every_n_epochs: Set a checkpoint for every n epochs
    :param save_top_k:
    :param profiler:
    :param gradient_clip_val:
    :param is_check:

中文config注释：
    @必填
    :param model_name:
    :param dataset_name: 其目录下包含数据集文件夹, 标签文件夹, 全部数据的命名list
    :param stage: 表示处于训练阶段还是测试阶段, fit表示训练, test表示测试
    :param max_epochs:
    :param batch_size:
    @可选
    :param version_info:
    :param accelerator:
    :param devices:
    :param accumulate_grad_batches:
    :param k_fold:
    :param kth_fold_start: 从第几个fold开始. 若使用重载训练, 则kth_fold_start为重载第几个fold, 第一个值为0.
                           非重载训练的情况下, 可以通过调整该值控制训练的次数;
    :param precision: 训练精度, 正常精度为32, 半精度为16. 精度代表每个参数的类型所占的位数.双精度（64、'64'或'64-true'）、
                      全精度（32、'32'或'32-true'），16位混合精度（16，'16'，'16-mixed'）或bfloat16混合精度（'bf16'，
                      'bf16-mixed'）。可用于CPU、GPU、TPU、HPU或IPU。
    :param log_name: 
    :param version_nth: 该值为重载训练的版本数或测试开始的版本数
    :param seed:
    :param path_final_save: 每次更新ckpt文件后, 将其存放到另一个位置
    :param every_n_epochs: 每n个epoch设置一个检查点
    :param save_top_k:
    :param profiler:
    :param gradient_clip_val:
    :param is_check:
"""
default_config = {
    'version_info': '',
    'accelerator': "auto",
    'devices': 1,
    'accumulate_grad_batches': 1,
    'k_fold': 1,
    'kth_fold_start': 0,
    'precision': '32-true',
    'log_name': 'lightning_logs',
    'version_nth': None,
    'seed': None,
    'path_final_save': None,
    'every_n_epochs': 1,
    'save_top_k': 1,
    'profiler': None,
    'gradient_clip_val': None,
    'is_check': False
}


def main(config):
    # Setting default config
    # 配置默认config
    config = {**default_config, **config}
    # Automatic setting config
    # 自动配置参数
    for kth_fold in range(config['kth_fold_start'], config['k_fold']):
        print(f'amount of fold is {kth_fold}|fold的数量为{kth_fold}')
        config['kth_fold'] = kth_fold
        config['time'] = str(datetime.now())
        if config['version_nth'] is not None:
            load_checkpoint_path = get_ckpt_path(config['version_nth'], config['log_name'])
        else:
            load_checkpoint_path = None
        logger = pl_loggers.TensorBoardLogger('logs/', name=config['log_name'])
        dm = DataModule(num_workers=min([cpu_count(), 8]), config=config)
        # SaveCheckpoint should be created before TrainModule to ensure the deterministic initialization of network
        # parameters.
        # SaveCheckpoint的创建需要在TrainModule之前, 以保证网络参数初始化的确定性
        save_checkpoint = SaveCheckpoint(config=config)
        imported = importlib.import_module('network.%(model_name)s' % config)
        if config['stage'] == 'fit':
            training_module = imported.TrainModule(config=config)
            trainer = pl.Trainer(logger=logger, precision=config['precision'], callbacks=[save_checkpoint],
                                 accelerator=config['accelerator'],
                                 devices=find_usable_cuda_devices(config['devices']) if torch.cuda.is_available() else
                                 config['devices'],
                                 max_epochs=config['max_epochs'], log_every_n_steps=1,
                                 accumulate_grad_batches=config['accumulate_grad_batches'],
                                 profiler=config['profiler'],
                                 gradient_clip_val=config['gradient_clip_val'],
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
                training_module = imported.TrainModule.load_from_checkpoint(
                    checkpoint_path=load_checkpoint_path,
                    **{'config': config})
                trainer = pl.Trainer(logger=logger, precision=config['precision'], callbacks=[save_checkpoint],
                                     profiler=config['profiler'],
                                     accelerator=config['accelerator'],
                                     devices=find_usable_cuda_devices(
                                         config['devices']) if torch.cuda.is_available() else config['devices'],
                                     )
                trainer.test(training_module, datamodule=dm)
        # The result can be viewed using ’tensorboard --logdir logs‘ in CMD, with the % prefix required
        # in Jupyter format
        # 在cmd中使用tensorboard --logdir logs命令可以查看结果，在Jupyter格式下需要加%前缀
        if config['version_nth'] is not None:
            if config['stage'] == 'fit':
                config['version_nth'] = None
            elif config['stage'] == 'test':
                config['version_nth'] += 1


class TerminalOutput:
    def __init__(self):
        self.stderr = sys.stderr  # 保存原始的 sys.stdout
        self.KeyboardInterrupt = False  # 用于存储捕获的输出内容

    def write(self, text):
        if 'KeyboardInterrupt' in text:
            self.KeyboardInterrupt = True
        self.stderr.write(text)  # 在终端上显示输出

    def flush(self):
        self.stderr.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-nth', type=str, help='task_nth. example format: tasks1', default='')
    args = parser.parse_args()

    # 创建自定义输出流对象
    terminal_output = TerminalOutput()
    # 将自定义输出流设置为 sys.stdout
    sys.stderr = terminal_output

    nth_thread = args.nth
    while True:
        # Obtain all parameters
        # 获得全部参数
        with open(f"./tasks{nth_thread}.json", "r", encoding='UTF-8') as f:
            configs = json.load(f)
        if len(configs) == 0:
            print('over|结束')
            break
        current_key = str(min([int(i) for i in list(configs.keys())]))
        print(f'Current_key is {current_key}')
        config = configs[current_key]
        main(config=config)
        if terminal_output.KeyboardInterrupt:
            break
        else:
            with open(f"./tasks{nth_thread}.json", "r", encoding='UTF-8') as f:
                configs = json.load(f)
            del configs[current_key]
            with open(f"./tasks{nth_thread}.json", "w", encoding='UTF-8') as f:
                f.write(json.dumps(configs, indent=2, ensure_ascii=False))
