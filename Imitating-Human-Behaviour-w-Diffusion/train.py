from argparse import ArgumentParser
import os
from collections import OrderedDict
from itertools import product

import pickle
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision import transforms
import torch.functional as F

from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity

from models import (
    Model_Cond_Diffusion,
    Model_mlp_diff,
    Model_cnn_mlp
)

DATASET_PATH = 'dataset'
SAVE_DATA_DIR = "output" 

EXPERIMENTS = [
    {
        "exp_name": "diffusion",
        "model_type": "diffusion",
        "drop_prob": 0.0,
    },
]

EXTRA_DIFFUSION_STEPS = [0, 2, 4, 8, 16, 32]
GUIDE_WEIGHTS = [0.0, 4.0, 8.0]

n_epoch = 100
lrate = 1e-4
device = "cuda"
n_hidden = 512
batch_size = 32
n_T = 50
net_type = "transformer"

class ClawCustomDataset(Dataset):
    def __init__(
        self, DATASET_PATH, transform=None, train_or_test="train", train_prop=0.90
    ):
        self.DATASET_PATH = DATASET_PATH
        # just load it all into RAM
        self.image_all = np.load(os.path.join(DATASET_PATH, "images_small.npy"), allow_pickle=True)
        self.image_all_large = np.load(os.path.join(DATASET_PATH, "images.npy"), allow_pickle=True)
        self.label_all = np.load(os.path.join(DATASET_PATH, "labels.npy"), allow_pickle=True)
        self.action_all = np.load(os.path.join(DATASET_PATH, "actions.npy"), allow_pickle=True)
        self.transform = transform
        n_train = int(self.image_all.shape[0] * train_prop)
        
        if train_or_test == "train":
            self.image_all = self.image_all[:n_train]
            self.label_all = self.label_all[:n_train]
            self.action_all = self.action_all[:n_train]
        elif train_or_test == "test":
            self.image_all = self.image_all[n_train:]
            self.label_all = self.label_all[n_train:]
            self.action_all = self.action_all[n_train:]
        else:
            raise NotImplementedError
        
        # normalize actions and images to range [0, 1]
        self.action_all = self.action_all / 64.0
        self.image_all = self.image_all / 255.0
        
    def __len__(self):
        return self.image_all.shape[0]
    
    def __getitem__(self, index):
        image = self.image_all[index]
        action = self.action_all[index]
        if self.transform:
            image = self.transform(image)
        return (image, action)
    
def train_claw(experiment, n_epoch, lrate, device, n_hidden, batch_size, n_T, net_type, EXTRA_DIFFUSION_STEPS, GUIDE_WEIGHTS):
    # Unpack experiment settings
    exp_name = experiment["exp_name"]
    model_type = experiment["model_type"]
    drop_prob = experiment["drop_prob"]
    
    # get datasets set up
    tf = transforms.Compose([])
    torch_data_train = ClawCustomDataset(
        DATASET_PATH, transform=tf, train_or_test="train", train_prop=0.90
    )
    dataload_train = DataLoader(
        torch_data_train, batch_size=batch_size, shuffle=True, num_workers=0
    )
    
    x_shape = torch_data_train.image_all.shape[1:]
    y_dim = torch_data_train.action_all.shape[1]
    
    # EBM langevin requires gradient for sampling
    requires_grad_for_eval = False
    
    # create model
    if model_type == "diffusion":
        nn_model = Model_cnn_mlp(
            x_shape, n_hidden, y_dim, embed_dim=128, net_type=net_type
        ).to(device)
        model = Model_Cond_Diffusion(
            nn_model,
            betas=(1e-4, 0.02),
            n_T=n_T,
            device=device,
            x_dim=x_shape,
            y_dim=y_dim,
            drop_prob=drop_prob,
            guide_w=0.0,
        )
    else:
        raise NotImplementedError
    
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lrate)
    
    for ep in tqdm(range(n_epoch), desc="Epoch"):
        results_ep = [ep]
        model.train()
        
        # lrate decay
        optim.param_groups[0]["lr"] = lrate * ((np.cos((ep / n_epoch) * np.pi) + 1) / 2)
        
        # train loop
        pbar = tqdm(dataload_train)
        loss_ep, n_batch = 0, 0
        for x_batch, y_batch in pbar:
            x_batch = x_batch.type(torch.FloatTensor).to(device)
            y_batch = y_batch.type(torch.FloatTensor).to(device)
            loss = model.loss_on_batch(x_batch, y_batch)
            optim.zero_grad()
            loss.backward()
            loss_ep += loss.detach().item()
            n_batch += 1
            pbar.set_description(f"train loss : {loss_ep / n_batch:.4f}")
            optim.step()
        results_ep.append(loss_ep / n_batch)
        
    model.eval()
    idxs = [14, 2, 0, 9, 5, 35, 16]
    extra_diffusion_steps = EXTRA_DIFFUSION_STEPS if exp_name == "diffusion" else [0]
    use_kdes = [False, True] if exp_name == "diffusion" else [False]
    guide_weight_list = GUIDE_WEIGHTS if exp_name == "cfg" else [None]
    idxs_data = [[] for _ in range(len(idxs))]
    for extra_diffusion_step, guide_weight, use_kde in product(extra_diffusion_steps, guide_weight_list, use_kdes):
        if extra_diffusion_step != 0 and use_kde:
            continue
        for i, idx in enumerate(idxs):
            x_eval = (
                torch.Tensor(torch_data_train.image_all[idx]).type(torch.FloatTensor).to(device)
            )
            x_eval_large = torch_data_train.image_all_large[idx]
            obj_mask_eval = torch_data_train.label_all[idx]
            if i == 0:
                obj_mask_eval_marginal = np.zeros_like(obj_mask_eval)
            obj_mask_eval_marginal += obj_mask_eval
            for j in range(6 if not use_kde else 300):
                x_eval_ = x_eval.repeat(50, 1, 1, 1)
                with torch.set_grad_enabled(requires_grad_for_eval):
                    if exp_name == "cfg":
                        model.guide_w = guide_weight
                    if model_type != "diffusion":
                        y_pred_ = model.sample(x_eval_).detach().cpu().numpy()
                    else:
                        if extra_diffusion_step == 0:
                            y_pred_ = (
                                model.sample(x_eval_, extract_embedding=True).detach().cpu().numpy()
                            )
                        
                        if use_kde:
                            # kde
                            torch_obs_many = x_eval_
                            action_pred_many = model.sample(torch_obs_many).cpu().numpy()
                            # fit kde to the sampled actions
                            kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(action_pred_many)
                            # choose the max likelihood one
                            log_density = kde.score_samples(action_pred_many)
                            idx = np.argmax(log_density)
                            y_pred_ = action_pred_many[idx][None, :]
                        else:
                            y_pred_ = model.sample_extra(x_eval_, extra_steps=extra_diffusion_step).detach().cpu().numpy()
                if j == 0:
                    y_pred_ = y_pred_
                else:
                    y_pred = np.concatenate([y_pred, y_pred_])
            x_eval = x_eval.detach().cpu().numpy()
            
            idxs_data[i] = {
                "idx" : idx,
                "x_eval_large" : x_eval_large,
                "obj_mask_eval" : obj_mask_eval,
                "y_pred" : y_pred,
            }
    alphas_bar = torch.cumprod()
    
   
    
        

if __name__ == "__main__":
    os.makedirs(SAVE_DATA_DIR, exist_ok=True)
    for experiment in EXPERIMENTS:
        train_claw(experiment, n_epoch, lrate, device, n_hidden, batch_size, n_T, net_type, EXTRA_DIFFUSION_STEPS, GUIDE_WEIGHTS)