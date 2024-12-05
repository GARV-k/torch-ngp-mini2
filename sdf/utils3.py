import os
import glob
import tqdm
import random
import warnings
import tensorboardX

import numpy as np
import pandas as pd
import math
import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

import trimesh
import mcubes
from rich.console import Console
from torch_ema import ExponentialMovingAverage

import packaging



def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if packaging.version.parse(torch.__version__) < packaging.version.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


def extract_fields(bound_min, bound_max, resolution, query_func, model):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3]
                    val = query_func(pts, model).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy() # [N, 1] --> [x, y, z]
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func, model):
    #print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func, model)

    #print(u.shape, u.max(), u.min(), np.percentile(u, 50))
    
    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles



class Trainer(object):
    def __init__(self, 
                 name,
                 n,# name of this experiment
                 models,
                 w,# network 
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 enc_optimizer=None,
                 net_optimizer=None,
                 w_optimizer=None, #optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metirc
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                ):
        
        self.name = name
        self.n = n
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()
        # self.w = w.to(self.device)
        self.w =w
        for model in models:
            model.to(self.device)
            if self.world_size > 1:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.models = models

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion
        self.enc_optimizers = []
        self.net_optimizer = net_optimizer(self.models[0])
        self.w_optimizer = w_optimizer(self.w)
        for idx in range(self.n):
            temp_opt = enc_optimizer(self.models[idx])
            self.enc_optimizers.append(temp_opt)
            
        self.enc_lr_schedulers = []   
        for optimizer in self.enc_optimizers:
            if lr_scheduler is None:
                lr_scheduler1 = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
            else:
                lr_scheduler1 = lr_scheduler(optimizer)
            self.enc_lr_schedulers.append(lr_scheduler1)
        
        if lr_scheduler is None:
            self.net_lr_scheduler = optim.lr_scheduler.LambdaLR(self.net_optimizer, lr_lambda=lambda epoch: 1) # fake scheduler   
        else:
            self.net_lr_scheduler = lr_scheduler(self.net_optimizer)       

        
        if lr_scheduler is None:
                self.w_lr_scheduler = optim.lr_scheduler.LambdaLR(self.w_optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
                self.w_lr_scheduler = lr_scheduler(self.w_optimizer)
        
        
        self.emas= []
        if ema_decay is not None:
            for model in self.models:
                ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)
                self.emas.append(ema)
        else:
            self.ema = None
        
        
        if ema_decay is not None:
                ema = ExponentialMovingAverage([self.w], decay=ema_decay)
                self.w_ema=ema
        else:
            self.w_ema = None
        
        
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
            }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth.tar"
            os.makedirs(self.ckpt_path, exist_ok=True)
            
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in models[0].parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)
        
        if torch.isinf(self.w_optimizer.param_groups[0]['params'][0]).any() or torch.isnan(self.w_optimizer.param_groups[0]['params'][0]).any():
            print("Inf or NaN detected in optimizer parameters")

    def __del__(self):
        if self.log_ptr: 
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute: 
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    ### ------------------------------	

    def train_step(self, data, model):
        # assert batch_size == 1
        X = data["points"][0] # [B, 3]
        y = data["sdfs"][0] # [B]
        
        #pred = self.model(X)
        pred = model(X)
        loss = self.criterion(pred, y)

        return pred, y, loss
    
    def w_train_step(self, data, model,w):
        # assert batch_size == 1
        # model.encoder.params = torch.nn.Parameter((model.encoder.params*w.params)+1e-6)
        X = data["points"][0] # [B, 3]
        y = data["sdfs"][0] # [B]
        
        #pred = self.model(X)
        pred = model(X,w)
        loss = self.criterion(pred, y)

        return pred, y, loss

    def eval_step(self, data,model):
        return self.train_step(data,model)

    def test_step(self, data,model):  
        X = data["points"][0]
        pred = model(X)
        return pred        

    def save_mesh(self, save_path=None, resolution=256):
        for j, model in enumerate(self.models):
            if save_path is None:
                save_path = os.path.join(self.workspace, 'validation', f'{self.name}_{self.epoch}_{j}.obj')
            else: 
                save_path = save_path[:-4]+f'_{j}.obj'
            self.log(f"==> Saving mesh to {save_path}")

            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            def query_func(pts,model):
                pts = pts.to(self.device)
                with torch.no_grad():   
                    with torch.cuda.amp.autocast(enabled=self.fp16):
                        sdfs = model(pts) #TODO
                return sdfs

            bounds_min = torch.FloatTensor([-1, -1, -1])
            bounds_max = torch.FloatTensor([1, 1, 1])
            
            vertices, triangles = extract_geometry(bounds_min, bounds_max, resolution=resolution, threshold=0, query_func=query_func, model=model)
            mesh = trimesh.Trimesh(vertices, triangles, process=False)
            mesh.export(save_path)
        # vertices, triangles = extract_geometry(bounds_min, bounds_max, resolution=resolution, threshold=0, query_func=query_func)

        # mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
        # mesh.export(save_path)

        self.log(f"==> Finished saving mesh.")

    ### ------------------------------

    def train(self, train_loaders, valid_loaders, max_epochs,macro_batch_size):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))
        
        for epoch in range(self.epoch + 1, max_epochs + 1):
        
                self.epoch = epoch

                self.train_one_epoch(train_loaders,self.models,macro_batch_size)
                if epoch%5==0:
                    for idx,model in enumerate(self.models):
                            torch.save(model.encoder,self.workspace+f'/GT_enc{idx+10}_{7}_{8}_12.pth')
                            torch.save(model.backbone,self.workspace+f'/GT_mlp_{7}_{8}_12.pth')
                            print("Completed saving path of models")
                torch.cuda.empty_cache()
                
            

        if self.use_tensorboardX and self.local_rank == 0:
                self.writer.close()
    
    def w_train(self, train_loaders, valid_loaders, max_epochs,macro_batch_size):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))
        
        for epoch in range(self.epoch + 1, max_epochs + 1):
        
                self.epoch = epoch

                self.w_train_one_epoch(train_loaders,self.models,macro_batch_size)
                if epoch%5==0:
                    for idx,model in enumerate(self.models):
                            torch.save(model.encoder,self.workspace+f'/GT_enc{idx+10}_{7}_{8}_12.pth')
                            torch.save(model.backbone,self.workspace+f'/GT_mlp_{7}_{8}_12.pth')
                            print("Completed saving path of models")
                torch.cuda.empty_cache()
            

        if self.use_tensorboardX and self.local_rank == 0:
                self.writer.close()
                
    

    def evaluate(self, loader):
        #if os.path.exists(self.best_path):
        #    self.load_checkpoint(self.best_path)
        #else:
        #    self.load_checkpoint()
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader)
        self.use_tensorboardX = use_tensorboardX



    def prepare_data(self, data):
        if isinstance(data, list):
            for i, v in enumerate(data):
                if isinstance(v, np.ndarray):
                    data[i] = torch.from_numpy(v).to(self.device, non_blocking=True)
                if torch.is_tensor(v):
                    data[i] = v.to(self.device, non_blocking=True)
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    data[k] = torch.from_numpy(v).to(self.device, non_blocking=True)
                if torch.is_tensor(v):
                    data[k] = v.to(self.device, non_blocking=True)
        elif isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(self.device, non_blocking=True)
        else: # is_tensor, or other similar objects that has `to`
            data = data.to(self.device, non_blocking=True)

        return data

    def forward_epoch_gradaccum(self,subset_loaders_iter, subset_models, div_factor):
        global total_loss
        global loader_epoch_loss
        for j, loader_iter in enumerate(subset_loaders_iter):
                data = next(loader_iter)
                #model = self.models[j]
                data = self.prepare_data(data)
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, truths, loss = self.train_step(data,subset_models[j])
                
                loss = loss/div_factor
                self.scaler.scale(loss).backward()
                loss_val = loss.item()
                
                total_loss += loss_val
                loader_epoch_loss += loss_val
                # del data, preds, truths,loss
                # torch.cuda.empty_cache()
                
    def w_forward_epoch_gradaccum(self,subset_loaders_iter, subset_models, div_factor):
        global total_loss
        global loader_epoch_loss
        for j, loader_iter in enumerate(subset_loaders_iter):
                data = next(loader_iter)
                #model = self.models[j]
                data = self.prepare_data(data)
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, truths, loss = self.w_train_step(data,subset_models[j],self.w)
                
                loss = loss/div_factor
                self.scaler.scale(loss).backward()
                loss_val = loss.item()
                
                total_loss += loss_val
                loader_epoch_loss += loss_val
                del data, preds, truths,loss
                torch.cuda.empty_cache()
    
    def train_one_epoch(self, loaders, models, macro_batch_size):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.enc_optimizers[0].param_groups[0]['lr']:.6f} ...")

        global total_loss
        total_loss = 0
        
        div_factor = math.ceil(len(models)/macro_batch_size)
        
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        for model in models:
            model.train()

        if self.world_size > 1:
            for loader in loaders:
                loader.sampler.set_epoch(self.epoch)
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loaders[0]) * loaders[0].batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
                
        self.local_step = 0
        N = min([len(loader) for loader in loaders])
        loaders_iter = [iter(loader) for loader in loaders]
        for i in range(N):
            global loader_epoch_loss
            loader_epoch_loss = 0
            self.local_step += 1
            self.global_step += 1
            
            for enc_opt in self.enc_optimizers:
                enc_opt.zero_grad()
            self.net_optimizer.zero_grad()
                
            subset_n = macro_batch_size
            for i in range(0, len(models), subset_n):
                subset_models = models[i:i + subset_n]
                subset_loaders_iter = loaders_iter[i:i + subset_n]
                self.forward_epoch_gradaccum(subset_loaders_iter, subset_models,div_factor)
                
            for enc_opt in self.enc_optimizers:
                self.scaler.step(enc_opt)
            self.scaler.step(self.net_optimizer)

            self.scaler.update()    

            for idx,_ in enumerate(self.models):
                if self.emas[idx] is not None:
                    self.emas[idx].update() 

            # if self.scheduler_update_every_step:
            #     [lr_scheduler.step() for lr_scheduler in self.enc_lr_schedulers]
            #     self.net_lr_scheduler.step()


            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)
                        
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loader_epoch_loss, self.global_step)
                    self.writer.add_scalar("train/lr", self.enc_optimizers[0].param_groups[0]['lr'], self.global_step)  #IDC (optimizer)

                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loader_epoch_loss:.4f} ({total_loss/self.local_step:.4f}), lr={self.enc_optimizers[0].param_groups[0]['lr']:.6f}") #IDC (optimizer)
                else:
                    pbar.set_description(f"loss={loader_epoch_loss:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loaders[0].batch_size)
        
        if self.scheduler_update_every_step:
            [lr_scheduler.step() for lr_scheduler in self.enc_lr_schedulers]
            self.net_lr_scheduler.step()
           
        torch.cuda.empty_cache()
        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.enc_lr_schedulers[0], torch.optim.lr_scheduler.ReduceLROnPlateau):
                [lr_scheduler.step(average_loss) for lr_scheduler in self.enc_lr_schedulers]
                self.net_lr_scheduler.step(average_loss)
            else:
                [lr_scheduler.step(average_loss) for lr_scheduler in self.enc_lr_schedulers]
                self.net_lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.") 
        
        
    def w_train_one_epoch(self, loaders, models, macro_batch_size):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.w_optimizer.param_groups[0]['lr']:.6f} ...")

        global total_loss
        total_loss = 0
        
        div_factor = math.ceil(len(models)/macro_batch_size)
        
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        for model in models:
            model.train()

        if self.world_size > 1:
            for loader in loaders:
                loader.sampler.set_epoch(self.epoch)
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loaders[0]) * loaders[0].batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
                
        self.local_step = 0
        N = min([len(loader) for loader in loaders])
        loaders_iter = [iter(loader) for loader in loaders]
        for i in range(N):
            global loader_epoch_loss
            loader_epoch_loss = 0
            self.local_step += 1
            self.global_step += 1
            
            for enc_opt in self.enc_optimizers:
                enc_opt.zero_grad()
            self.net_optimizer.zero_grad()
            self.w_optimizer.zero_grad()
                
            subset_n = macro_batch_size
            for i in range(0, len(models), subset_n):
                subset_models = models[i:i + subset_n]
                subset_loaders_iter = loaders_iter[i:i + subset_n]
                self.w_forward_epoch_gradaccum(subset_loaders_iter, subset_models,div_factor)
                
            # for enc_opt in self.enc_optimizers:
            #     self.scaler.step(enc_opt)
            # self.scaler.step(self.net_optimizer)
            self.scaler.step(self.w_optimizer)

            self.scaler.update()

            for idx,_ in enumerate(self.models):
                if self.emas[idx] is not None:
                    self.emas[idx].update() 
            if self.w_ema is not None:
                    self.w_ema.update()

            # if self.scheduler_update_every_step:
            #     # [lr_scheduler.step() for lr_scheduler in self.enc_lr_scheduler]
            #     # self.net_lr_scheduler.step()
            #     self.w_lr_scheduler.step()
                


            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)
                        
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loader_epoch_loss, self.global_step)
                    self.writer.add_scalar("train/lr", self.enc_optimizers[0].param_groups[0]['lr'], self.global_step)  #IDC (optimizer)

                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loader_epoch_loss:.4f} ({total_loss/self.local_step:.4f}), lr={self.enc_optimizers[0].param_groups[0]['lr']:.6f}") #IDC (optimizer)
                else:
                    pbar.set_description(f"loss={loader_epoch_loss:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loaders[0].batch_size)
           
        if self.scheduler_update_every_step:
                # [lr_scheduler.step() for lr_scheduler in self.enc_lr_scheduler]
                # self.net_lr_scheduler.step()
                self.w_lr_scheduler.step()
        
        torch.cuda.empty_cache()
        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.enc_lr_schedulers[0], torch.optim.lr_scheduler.ReduceLROnPlateau):
                # [lr_scheduler.step(average_loss) for lr_scheduler in self.enc_lr_schedulers]
                # self.net_lr_scheduler.step(average_loss)
                self.w_lr_scheduler.step(average_loss)
            else:
                # [lr_scheduler.step(average_loss) for lr_scheduler in self.enc_lr_schedulers]
                # self.net_lr_scheduler.step()
                self.w_lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")   


    def evaluate_one_epoch(self, loaders):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        [model.eval() for model in self.models]

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loaders[0]) * loaders[0].batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0
            N = min([len(loader) for loader in loaders])
            loaders_iter = [iter(loader) for loader in loaders]
            for i in range(N):
                self.local_step += 1
                    
                for j, loader_iter in enumerate(loaders_iter):
                    if self.emas[j] is not None:  #TODO
                        self.emas[j].store()
                        self.emas[j].copy_to()
                    
                    data = next(loader_iter)
                    data = self.prepare_data(data)
                    with torch.cuda.amp.autocast(enabled=self.fp16):
                        preds, truths, loss = self.eval_step(data,self.models[j])
                    
                    if self.emas[j] is not None:  
                        self.emas[j].restore()
                    
                    if self.world_size > 1:
                        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                        loss = loss / self.world_size
                        
                        preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                        dist.all_gather(preds_list, preds)
                        preds = torch.cat(preds_list, dim=0)

                        truths_list = [torch.zeros_like(truths).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                        dist.all_gather(truths_list, truths)
                        truths = torch.cat(truths_list, dim=0)

                    loss_val = loss.item()
                    total_loss += loss_val

                    # only rank = 0 will perform evaluation.
                    if self.local_rank == 0:

                        for metric in self.metrics:
                            metric.update(preds, truths)

                        pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                        pbar.update(loaders[j].batch_size)

            
            # self.local_step = 0
            # for data in loader:    
            #     self.local_step += 1
                
            #     data = self.prepare_data(data)

            #     if self.ema is not None:  #TODO
            #         self.ema.store()
            #         self.ema.copy_to()
            
            #     with torch.cuda.amp.autocast(enabled=self.fp16):
            #         preds, truths, loss = self.eval_step(data)

            #     if self.ema is not None:   #TODO
            #         self.ema.restore()
                
            #     # all_gather/reduce the statistics (NCCL only support all_*)
            #     if self.world_size > 1:
            #         dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            #         loss = loss / self.world_size
                    
            #         preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
            #         dist.all_gather(preds_list, preds)
            #         preds = torch.cat(preds_list, dim=0)

            #         truths_list = [torch.zeros_like(truths).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
            #         dist.all_gather(truths_list, truths)
            #         truths = torch.cat(truths_list, dim=0)

            #     loss_val = loss.item()
            #     total_loss += loss_val

            #     # only rank = 0 will perform evaluation.
            #     if self.local_rank == 0:

            #         for metric in self.metrics:
            #             metric.update(preds, truths)

            #         pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
            #         pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
            else:
                self.stats["results"].append(average_loss) # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, full=False, best=False):

        state = {
            'epoch': self.epoch,
            'stats': self.stats,
        }

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()
        
        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{self.name}_ep{self.epoch:04d}.pth.tar"

            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = self.stats["checkpoints"].pop(0)
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

            torch.save(state, file_path)

        else:    
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results 
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()
                    
                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")
            
    def load_checkpoint(self, checkpoint=None):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth.tar'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        
        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")            

        if self.ema is not None and 'ema' in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict['ema'])

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        
        if self.optimizer and  'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer, use default.")
        
        # strange bug: keyerror 'lr_lambdas'
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler, use default.")

        if 'scaler' in checkpoint_dict:
            self.scaler.load_state_dict(checkpoint_dict['scaler'])                
