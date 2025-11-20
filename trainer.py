import collections
import os
import math
from time import time
from tqdm import tqdm
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from recbole.trainer import Trainer
from recbole.utils import set_color, get_gpu_usage, EvaluatorType, early_stopping, dict2str, get_local_time
from recbole.data.dataloader import FullSortEvalDataLoader

from collections import defaultdict
import torch.distributed as dist
class DecoRecTrainer(Trainer):
    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False,is_fixed=False):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.
            show_progress (bool): Show the progress of training epoch. Defaults to ``False``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, it will return a
            tuple which includes the sum of loss in each part.
        """
        if is_fixed==1:
            for param in self.model.text_adaptor.parameters():
                param.requires_grad = False
            for param in self.model.img_adaptor.parameters():
                param.requires_grad = False
        elif is_fixed==2:
            param_dict = collections.defaultdict(list)
            for name, param in self.model.named_parameters():
                if "embedding" in name:
                    param_dict[name].append(param)
                else:
                    param_dict["other"].append(param)
            optimizer = torch.optim.Adam([
                {'params': param_dict["item_embedding"], 'lr': 1e-4},
                {'params': param_dict["item_embedding_img"], 'lr': 1e-4},

                {'params': param_dict['other'], 'lr': 1e-3}
            ])
            self.optimizer = optimizer


        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
            ) if show_progress else train_data
        )
        param_dict = collections.defaultdict(list)
        for name, param in self.model.named_parameters():
            if "embedding" in name:
                param_dict[name].append(param)
            else:
                param_dict["other"].append(param)
        optimizer = torch.optim.Adam([
            {'params': param_dict["item_embedding"], 'lr': 1e-4},
            {'params': param_dict["item_embedding_img"], 'lr': 1e-4},

            {'params': param_dict['other'], 'lr': 1e-3}
        ])
        self.optimizer = optimizer

        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            losses = loss_func(interaction)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
        return total_loss

    def fit(
            self,
            train_data,
            valid_data=None,
            verbose=True,
            saved=True,
            show_progress=False,
            callback_fn=None,
    ):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True
            show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
            callback_fn (callable): Optional callback function executed at end of epoch.
                                    Includes (epoch_idx, valid_score) input arguments.

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1, verbose=verbose)

        self.eval_collector.data_collect(train_data)
        if self.config["train_neg_sample_args"].get("dynamic", False):
            train_data.get_model(self.model)
        valid_step = 0

        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(
                train_data, epoch_idx, show_progress=show_progress,is_fixed=self.is_fixed
            )
            self.train_loss_dict[epoch_idx] = (
                sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            )
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss
            )
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
            self.wandblogger.log_metrics(
                {"epoch": epoch_idx, "train_loss": train_loss, "train_step": epoch_idx},
                head="train",
            )

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(
                    valid_data, show_progress=show_progress
                )

                (
                    self.best_valid_score,
                    self.cur_step,
                    stop_flag,
                    update_flag,
                ) = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger,
                )
                valid_end_time = time()
                valid_score_output = (
                                             set_color("epoch %d evaluating", "green")
                                             + " ["
                                             + set_color("time", "blue")
                                             + ": %.2fs, "
                                             + set_color("valid_score", "blue")
                                             + ": %f]"
                                     ) % (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = (
                        set_color("valid result", "blue") + ": \n" + dict2str(valid_result)
                )
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar("Vaild_score", valid_score, epoch_idx)
                self.wandblogger.log_metrics(
                    {**valid_result, "valid_step": valid_step}, head="valid"
                )

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = "Finished training, best eval result in epoch %d" % (
                            epoch_idx - self.cur_step * self.eval_step
                    )
                    if verbose:
                        self.logger.info(stop_output)
                    break

                valid_step += 1

        self._add_hparam_to_tensorboard(self.best_valid_score)
        return self.best_valid_score, self.best_valid_result