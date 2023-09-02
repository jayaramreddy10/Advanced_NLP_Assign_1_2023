import os
import copy
import numpy as np
import torch
import einops
import pdb
import os
from timer import Timer
from arrays import batch_to_device, to_np, apply_dict, to_device
from torch import nn
import wandb
import time

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer_jayaram(object):  
    def __init__(
        self,
        model,
        train_dataset, 
        val_dataset, 
        is_LSTM = False, 
        ema_decay=0.995,
        train_batch_size=512,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=40000,
        save_parallel=False,
        # results_folder='/home/jayaram/research/research_tracks/table_top_rearragement/test_diffusion_planning/logs/maze2d-test/diffusion',
        results_folder='/home/jayaram/research/NLP_course_papers_and_content/Assignments/Assign_1/logs/NNLM',
        n_reference=50,
        n_samples=10,
        bucket=None,
    ):
        super().__init__()
        self.model = model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_dataloader = cycle(torch.utils.data.DataLoader(
            self.train_dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        )
        # self.dataloader_vis = cycle(torch.utils.data.DataLoader(
        #     self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        # ))
        self.optimizer = torch.optim.Adam(model.parameters(), lr=train_lr)

        if(is_LSTM):
            self.logdir = '/home/jayaram/research/NLP_course_papers_and_content/Assignments/Assign_1/logs/LSTM'
        else: 
            self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        self.n_samples = n_samples
        self.criterion = nn.CrossEntropyLoss()  # Mean Squared Error loss
        self.reset_parameters()
        self.step = 0

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train_NNLM(self, device, epoch_no, n_train_steps):
        isExist = os.path.exists(self.logdir)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(self.logdir)
            print("The new directory is created!")  
            
        timer = Timer()
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.train_dataloader)   

                # NNLM: batch[0].size: (B, window_size*embed_size)
                # LSTM: batch[0].size: (B, window_size, embed_size), we need to reshape this to (B, window_size, embed_size) to pass it to LSTM

                # batch[1] size: (B, )
                for i, el in enumerate(batch):
                    batch[i] = to_device(batch[i])
                
                outputs = self.model(batch[0])   #logits in this case (B, vocab_size) 
                _, label_predictions = torch.max(outputs, 1)
                targets = batch[1]
                targets = targets.to(torch.long)

                loss = self.criterion(outputs, targets)
                loss = loss / self.gradient_accumulate_every
                # for name, param in self.model.named_parameters():
                #     if param.grad is not None:
                #         print(f'Parameter: {name}, Gradient norm: {param.grad.norm()}')

                loss.backward()

            #scale back loss
            loss = loss * self.gradient_accumulate_every
            wandb.log({'train loss NNLM': loss, 'epoch': epoch_no, 'step no': step}) #, 'batch': t})

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.log_freq == 0:
                print(f'{self.step}: {loss:8.4f}  | t: {timer():8.4f}')
            self.step += 1

        label = epoch_no
        self.save(label)
        
        # report validation loss/scores
        validation_loss = 0.0    #for entire val dataset in current epoch
        # Set your model to evaluation mode
        self.model.eval()
        # Iterate through the validation dataset
        with torch.no_grad():
            for  idx, val_batch in enumerate(self.val_dataloader):
                for i, el in enumerate(val_batch):
                    val_batch[i] = to_device(val_batch[i])
                val_outputs = self.model(val_batch[0])
                _, val_predictions = torch.max(val_outputs, 1)
                val_targets = val_batch[1]
                val_targets = val_targets.to(torch.long)
                # Calculate validation loss
                val_loss = self.criterion(val_outputs, val_targets)
                validation_loss += val_loss            

        average_validation_loss = validation_loss / len(self.val_dataloader)
        print(f'Validation Loss: {average_validation_loss:.4f}, Epoch: {epoch_no}')
        wandb.log({'val loss NNLM': average_validation_loss, 'epoch': epoch_no}) #, 'batch': t})

        # Set your model back to training mode
        self.model.train()

    def train_LSTM(self, device, epoch_no, n_train_steps):
        isExist = os.path.exists(self.logdir)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(self.logdir)
            print("The new directory is created!")  
            
        timer = Timer()
        for step in range(n_train_steps):
            loss = 0.0
            for i in range(self.gradient_accumulate_every):
                batch = next(self.train_dataloader)   

                # NNLM: batch[0].size: (B, window_size*embed_size)
                # LSTM: batch[0].size: (B, window_size, embed_size), we need to reshape this to ( window_size, B, embed_size) to pass it to LSTM
                batch[0] = einops.rearrange(batch[0], 'b w e -> w b e') 

                # NNLM: batch[1].size: (B, )
                # LSTM: batch[1].size: (B, window_size/seq_len)
                for k, el in enumerate(batch):
                    batch[k] = to_device(batch[k])
                
                outputs, _ = self.model(batch[0])   #logits in this case (seq_len, B, vocab_size) 
                # outputs = einops.rearrange(outputs, 'w b e -> b w e')   #(B, seq_len, vocab_size)
                outputs = outputs.view(-1, outputs.size(-1))

                # _, label_predictions = torch.max(outputs, 1)
                targets = batch[1].T
                targets = targets.to(torch.long)   #(seq_len, B = 512)
                targets = targets.reshape(-1)

                loss = self.criterion(outputs, targets)
                loss = loss / self.gradient_accumulate_every
                
                # for name, param in self.model.named_parameters():
                #     if param.grad is not None:
                #         print(f'Parameter: {name}, Gradient norm: {param.grad.norm()}')

                loss.backward()

            #scale back loss
            loss = loss * self.gradient_accumulate_every
            wandb.log({'train loss LSTM': loss, 'epoch': epoch_no, 'step no': step}) #, 'batch': t})
            
            # train_perplexity = compute_perplexity()
            # wandb.log({'train_perplexity': train_perplexity, 'epoch': epoch_no, 'step no': step}) #, 'batch': t})
            
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.log_freq == 0:
                print(f'{self.step}: {loss:8.4f}  | t: {timer():8.4f}')
            self.step += 1

        label = epoch_no
        self.save(label)

        # report validation loss/scores
        validation_loss = 0.0    #for entire val dataset in current epoch
        # Set your model to evaluation mode
        self.model.eval()
        # Iterate through the validation dataset
        # print('val dataloader len: {}'.format(len(self.val_dataloader)))

        with torch.no_grad():
            for val_batch in self.val_dataloader:
                # time_s = time.time()

                val_batch[0] = einops.rearrange(val_batch[0], 'b w e -> w b e') 
                for i, el in enumerate(val_batch):
                    val_batch[i] = to_device(val_batch[i])
                val_outputs, _ = self.model(val_batch[0])
                val_outputs = val_outputs.view(-1, val_outputs.size(-1))
                # _, val_predictions = torch.max(val_outputs, 1)
                val_targets = val_batch[1]
                val_targets = val_targets.to(torch.long)
                val_targets = val_targets.reshape(-1)
                # Calculate validation loss
                val_loss = self.criterion(val_outputs, val_targets)
                validation_loss += val_loss.item()          

                # time_e = time.time()
                # print('one batch val time: {}'.format(time_e - time_s))

        
        average_validation_loss = validation_loss / len(self.val_dataloader)
        # log results to text file
        print(f'Validation Loss: {average_validation_loss:.4f}, Epoch: {epoch_no}')
        wandb.log({'val loss LSTM': average_validation_loss, 'epoch': epoch_no}) #, 'batch': t})

        # Set your model back to training mode
        self.model.train()

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        isExist = os.path.exists(self.logdir)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(self.logdir)
            print("The new directory is created!")        

        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}')

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])


