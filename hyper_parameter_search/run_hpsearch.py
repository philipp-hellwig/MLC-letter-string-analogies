"""
Run hyperparameter search over hyperparameters specified in config.py
"""

import argparse
import time
import math

import torch
from torch.utils.data import DataLoader
import numpy as np

from .. import datasets as dat
from .. import checkpoint
from ..evaluate import evaluate_ll, evaluate_predictions
from ..model import MLC
from ..train import train
from ..train_lib import timeSince
from config import hp_varied, hp_fixed

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Training on {DEVICE}.")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_model', type=str, default='models', help='Directory for saving model files')
    parser.add_argument('--dir_data', type=str, default='data', help='Directory for loading datasets')
    args = parser.parse_args()
    
    D_train = dat.LetterStringDataset(mode="train", data_dir="../data/base_probs")
    D_val =  dat.LetterStringDataset(mode="val", data_dir="../data/base_probs")
    val_dataloader = DataLoader(D_val, batch_size=5000)
    print(f"Training with datasets from directory {args.dir_data}")

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=D_train.langs['output'].PAD_idx)

    for bs in hp_varied["batch_size"]:
        train_dataloader = DataLoader(D_train, batch_size=bs, shuffle=True)
        
        for lr in hp_varied["learning_rate"]:
            optimizer = torch.optim.AdamW(model.parameters(),lr=lr, betas=(0.9,0.95), weight_decay=0.01)
            if hp_fixed["lr_warmup"]:
                print('    with LR warmup ON (1st epoch)')
                scheduler_epoch = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=hp_fixed["lr_end_factor"], total_iters=hp_fixed["nepochs"]-2)
                nstep_epoch_estimate = math.floor(len(D_train)/bs)
                scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-4, end_factor=1.0, total_iters=nstep_epoch_estimate)
            else:            
                print('    with LR warmup OFF')
                scheduler_epoch = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=hp_fixed["lr_end_factor"], total_iters=hp_fixed["nepochs"]-1)            

            for nheads in hp_varied["nheads"]:
                
                for dropout in hp_varied["dropout"]:
                    model = MLC(
                        hidden_size=hp_fixed["emb_size"], 
                        input_size=D_train.langs['input'].n_symbols, 
                        output_size=D_train.langs['output'].n_symbols,
                        nhead=nheads,
                        PAD_idx_input=D_train.langs['input'].PAD_idx, 
                        PAD_idx_output=D_train.langs['output'].PAD_idx,
                        nlayers_encoder=hp_fixed["nlayers_encoder"], 
                        nlayers_decoder=hp_fixed["nlayers_decoder"],
                        dropout_p=dropout,
                        activation= hp_fixed["activation"],
                        ff_mult=hp_fixed["ff_multiplier"]
                    )

                    hp_current = {
                        "batch_size": bs,
                        "learning_rate": lr,
                        "nheads": nheads,
                        "dropout": dropout
                    }
                    execute_training(
                        hp_current=hp_current,
                        model=model,
                        scheduler_warmup=scheduler_warmup,
                        scheduler_epoch=scheduler_epoch,
                        val_dataloader=val_dataloader,
                        train_dataloader=train_dataloader,
                        optimizer=optimizer,
                        loss_fn=loss_fn
                    )


def execute_training(
        hp_current: dict,
        model,
        scheduler_warmup,
        scheduler_epoch,
        val_dataloader,
        train_dataloader,
        optimizer,
        loss_fn
    ):

    best_val_loss = float('inf')
    counter = 0 # num updates since the loss was last reported
    step = 0
    train_tracker = []
    val_accuracy_by_epoch = []
    epoch_start = 1
    nsteps_estimate = math.ceil(hp_fixed["nepochs"]*len(train_dataloader.dataset)/hp_fixed["batch_size"])
    avg_train_loss = 0.
    start = time.time()
    save_path = "|".join([key + "=" + str(value).replace(".","_") for key,value in hp_current.items()]) + ".pt"
    print(f"Training with hyperparameters: {save_path.replace('.pt','')}")

    # training loop:
    for epoch in range(epoch_start,hp_fixed["nepochs"]+1):
        
        print("Epoch",epoch,"\n-------------------------------")
        for train_batch in train_dataloader:
            loss = train(train_batch, model, loss_fn, optimizer)
            avg_train_loss += loss
            counter += 1
            step += 1  
                        
            if step in [1,25] or step % 100 == 0:
                mylr = optimizer.param_groups[0]['lr']
                mytracker = {'epoch':epoch, 'step':step, 'lr':mylr, 'avg_train_loss':avg_train_loss/counter}
                print('{:s} ({:d} {:.0f}% finished) LR: {:.7f}, TrainLoss: {:.4f}, '.format(timeSince(start, float(step) / float(nsteps_estimate)),
                                        step, float(step) / float(nsteps_estimate) * 100., mylr, avg_train_loss/counter), end='')
                
                # compute validation loss
                total_ll, total_N = evaluate_ll(val_dataloader, model, val_dataloader.dataset.langs, loss_fn=loss_fn)
                val_loss = -total_ll / total_N
                print('ValLoss: {:.4f}'.format(val_loss))
                mytracker['val_loss'] = val_loss
                mytracker['val_acc'] = torch.nan
                
                # update best validation loss
                best_val_loss = val_loss if val_loss < best_val_loss else best_val_loss
                avg_train_loss = 0.
                counter = 0
                train_tracker.append(mytracker)
            
            # if warm-up, adjust learning rate for each step of the first epoch
            if hp_fixed["lr_warmup"] and epoch==1: 
                scheduler_warmup.step() 
        
        # after each epoch, calculate and save val accuracy:
        scores, trans_types = evaluate_predictions(val_dataloader, model, max_length=20, eval_type="max")
        val_accuracy = dict()
        val_accuracy["epoch"] = epoch
        val_accuracy["overall"] = np.mean(scores)
        for trans in np.unique(trans_types):
            val_accuracy[trans] = np.mean(scores[trans_types==trans])
        val_accuracy_by_epoch.append(val_accuracy)
        # after each epoch, adjust the general learning rate
        if epoch>1 or not hp_fixed["lr_warmup"]: 
            scheduler_epoch.step()
        checkpoint.save(save_path,step,epoch,model,optimizer,scheduler_epoch,train_tracker, val_accuracy_by_epoch, best_val_loss, hp_current)
    print('Training complete.')


if __name__=="__main__":
    main()