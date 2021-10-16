import os
import numpy.random
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import shutil
from pytorch_lightning.utilities import rank_zero_info
from utils import zip_dir
import re


class SaveCheckpoint(ModelCheckpoint):
    def __init__(self,
                 max_epochs,
                 seed=None,
                 every_n_epochs=None,
                 path_final_save=None,
                 monitor=None,
                 save_top_k=None,
                 verbose=False,
                 mode='min',
                 no_save_before_epoch=0):
        """
        通过回调实现checkpoint的保存逻辑, 同时具有回调函数中定义on_validation_end等功能.

        :param max_epochs:
        :param seed:
        :param every_n_epochs:
        :param path_final_save:
        :param monitor:
        :param save_top_k:
        :param verbose:
        :param mode:
        :param no_save_before_epoch:
        """
        super().__init__(every_n_epochs=every_n_epochs, verbose=verbose, mode=mode)
        self.mode = mode
        numpy.random.seed(seed)
        self.seeds = numpy.random.randint(0, 2000, max_epochs)
        pl.seed_everything(seed)
        self.path_final_save = path_final_save
        self.monitor = monitor
        self.save_top_k = save_top_k
        self.flag_sanity_check = 0
        self.no_save_before_epoch = no_save_before_epoch

    def on_validation_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """
        修改随机数逻辑,网络的随机种子给定,取样本的随机种子由给定的随机种子生成,保证即使重载训练每个epoch具有不同的抽样序列.
        同时保存checkpoint.

        :param trainer:
        :param pl_module:
        :return:
        """
        # 第一个epoch使用原始输入seed作为种子, 后续的epoch使用seeds中的第epoch-1个作为种子
        if self.flag_sanity_check == 0:
            self.flag_sanity_check = 1
        else:
            pl.seed_everything(self.seeds[trainer.current_epoch])
        super().on_validation_end(trainer, pl_module)

    def _save_top_k_checkpoint(self, trainer: 'pl.Trainer', monitor_candidates) -> None:
        epoch = monitor_candidates.get("epoch")
        if self.monitor is None or self.save_top_k == 0 or epoch < self.no_save_before_epoch:
            return

        current = monitor_candidates.get(self.monitor)

        if self.check_monitor_top_k(trainer, current):
            self._update_best_and_save(current, trainer, monitor_candidates)
            if self.mode == 'max':
                best_model_value = max([float(item) for item in list(self.best_k_models.values())])
            else:
                best_model_value = min([float(item) for item in list(self.best_k_models.values())])
            version_name = 'version_unkown'
            for item in re.split(r'[/|\\]', self.dirpath):
                if 'version_' in item:
                    version_name = item
                    break
            # 保存版本信息(准确率等)到txt中
            if not os.path.exists('./logs/default/version_info.txt'):
                with open('./logs/default/version_info.txt', 'w', encoding='utf-8') as f:
                    f.write(version_name + ' ' + str(best_model_value) + '\n')
            else:
                with open('./logs/default/version_info.txt', 'r', encoding='utf-8') as f:
                    info_list = f.readlines()
                info_list = [item.strip('\n').split(' ') for item in info_list]
                # 对list进行转置, 现在行为版本号和其数据, 列为不同的版本
                info_list = list(map(list, zip(*info_list)))
                if version_name in info_list[0]:
                    for cou in range(len(info_list[0])):
                        if version_name == info_list[0][cou]:
                            info_list[1][cou] = str(best_model_value)
                else:
                    info_list[0].append(version_name)
                    info_list[1].append(str(best_model_value))
                # 对list进行转置
                info_list = list(map(list, zip(*info_list)))
                with open('./logs/default/version_info.txt', 'w', encoding='utf-8') as f:
                    for line in info_list:
                        line = " ".join(line)
                        f.write(line + '\n')
            # 每次更新ckpt文件后, 将其存放到另一个位置
            if self.path_final_save is not None:
                zip_dir('./logs/default/' + version_name, './' + version_name + '.zip')
                if os.path.exists(self.path_final_save + '/' + version_name + '.zip'):
                    os.remove(self.path_final_save + '/' + version_name + '.zip')
                shutil.move('./' + version_name + '.zip', self.path_final_save)
        elif self.verbose:
            epoch = monitor_candidates.get("epoch")
            step = monitor_candidates.get("step")
            best_model_values = 'now best model:'
            for cou_best_model in self.best_k_models:
                best_model_values = ' '.join(
                    (best_model_values, str(round(float(self.best_k_models[cou_best_model]), 4))))
            rank_zero_info(
                f"\nEpoch {epoch:d}, global step {step:d}: {self.monitor} ({float(current):f}) was not in "
                f"top {self.save_top_k:d}({best_model_values:s})")
