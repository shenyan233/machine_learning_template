import os
import numpy.random
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import shutil
from pytorch_lightning.utilities import rank_zero_info
from utils import zip_dir


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
            if self.path_final_save is not None:
                zip_dir('./logs', './logs.zip')
                if os.path.exists(self.path_final_save + '/logs.zip'):
                    os.remove(self.path_final_save + '/logs.zip')
                shutil.move('./logs.zip', self.path_final_save)
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
