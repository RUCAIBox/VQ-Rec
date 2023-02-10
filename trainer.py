import os
from time import time
from tqdm import tqdm
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from recbole.trainer import Trainer
from recbole.utils import early_stopping, dict2str, set_color, get_gpu_usage
from collections import OrderedDict


class VQRecTrainer(Trainer):
    def __init__(self, config, model):
        super(VQRecTrainer, self).__init__(config, model)

        self.reassign_steps = config['reassign_steps']
        assert self.reassign_steps is not None
        
    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
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
        if self.config['train_neg_sample_args'].get('dynamic', 'none') != 'none':
            train_data.get_model(self.model)
        valid_step = 0

        for epoch_idx in range(self.start_epoch, self.epochs):
            #### Index Assignment Special
            if epoch_idx < self.reassign_steps:
                for _ in self.model.parameters():
                    _.requires_grad = False
                self.logger.info('Only index assignment tuned.')
                self.model.index_assignment_flag = True
                self.model.trans_matrix.requires_grad = True
            elif epoch_idx == self.reassign_steps:
                for _ in self.model.parameters():
                    _.requires_grad = True
                for _ in self.model.position_embedding.parameters():
                    _.requires_grad = False
                for _ in self.model.trm_encoder.parameters():
                    _.requires_grad = False
                
                self.model.trans_matrix.requires_grad = False
                self.model.reassigned_code_embedding = self.model.code_projection()
                embed_weight = self.model.reassigned_code_embedding.detach()
                embed_weight[0,:] = 0
                reassigned_state_dict = OrderedDict()
                reassigned_state_dict['weight'] = embed_weight
                self.model.pq_code_embedding.load_state_dict(reassigned_state_dict)
                self.model.index_assignment_flag = False
                self.logger.info(f'Fix index assignment, model tuned.')
            #### End Index Assignment
            
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
            self.wandblogger.log_metrics({'epoch': epoch_idx, 'train_loss': train_loss, 'train_step':epoch_idx}, head='train')

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data, show_progress=show_progress)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger
                )
                valid_end_time = time()
                valid_score_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue')
                                    + ": %.2fs, " + set_color("valid_score", 'blue') + ": %f]") % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = set_color('valid result', 'blue') + ': \n' + dict2str(valid_result)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar('Vaild_score', valid_score, epoch_idx)
                self.wandblogger.log_metrics({**valid_result, 'valid_step': valid_step}, head='valid')

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break

                valid_step+=1

        self._add_hparam_to_tensorboard(self.best_valid_score)
        return self.best_valid_score, self.best_valid_result
    
    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
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
        for batch_idx, interaction in enumerate(iter_data):
            #### Index Assignment Special
            if self.model.index_assignment_flag:
                self.model.reassigned_code_embedding = self.model.code_projection()
            #### End Index Assignment
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            losses = loss_func(interaction)
            if isinstance(losses, tuple):
                if epoch_idx < self.config['warm_up_step']:
                    losses = losses[:-1]
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
        return total_loss


class DDPPretrainTrainer(Trainer):
    def __init__(self, config, model):
        super(DDPPretrainTrainer, self).__init__(config, model)
        self.pretrain_epochs = self.config['pretrain_epochs']
        self.save_step = self.config['save_step']
        self.rank = config['rank']
        self.world_size = config['world_size']
        self.lrank = self._build_distribute(rank=self.rank, world_size=self.world_size)
        self.logger.info(f'Let\'s use {torch.cuda.device_count()} GPUs to train {self.config["model"]} ...')

    def _build_distribute(self, rank, world_size):
        from torch.nn.parallel import DistributedDataParallel
        # credit to @Juyong Jiang
        # 1 set backend
        torch.distributed.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        # 2 get distributed id
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device_dis = torch.device("cuda", local_rank)
        # 3, 4 assign model to be distributed
        self.model.to(device_dis)
        self.model.pq_codes = self.model.pq_codes.to(device_dis)
        self.model = DistributedDataParallel(self.model, 
                                             device_ids=[local_rank],
                                             output_device=local_rank).module
        return local_rank

    def save_pretrained_model(self, epoch, saved_model_file):
        r"""Store the model parameters information and training information.
        Args:
            epoch (int): the current epoch id
            saved_model_file (str): file name for saved pretrained model
        """
        state = {
            'config': self.config,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, saved_model_file)

    def _trans_dataload(self, interaction):
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler

        #using pytorch dataload to re-wrap dataset
        def sub_trans(dataset):
            dis_loader = DataLoader(dataset=dataset,
                                    batch_size=dataset.shape[0],
                                    sampler=DistributedSampler(dataset, shuffle=False))
            for data in dis_loader:
                batch_data = data

            return batch_data
        #change `interaction` datatype to a python `dict` object.  
        #for some methods, you may need transfer more data unit like the following way.  

        data_dict = {}
        for k, v in interaction.interaction.items():
            data_dict[k] = sub_trans(v)
        return data_dict

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
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
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            interaction = self._trans_dataload(interaction)
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
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
        return total_loss

    def pretrain(self, train_data, verbose=True, show_progress=False):
        for epoch_idx in range(self.start_epoch, self.pretrain_epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)

            if (epoch_idx + 1) % self.save_step == 0 and self.lrank == 0:
                saved_model_file = os.path.join(
                    self.checkpoint_dir,
                    '{}-{}-{}.pth'.format(self.config['model'], self.config['dataset'], str(epoch_idx + 1))
                )
                self.save_pretrained_model(epoch_idx, saved_model_file)
                update_output = set_color('Saving current', 'blue') + ': %s' % saved_model_file
                if verbose:
                    self.logger.info(update_output)

        return self.best_valid_score, self.best_valid_result
