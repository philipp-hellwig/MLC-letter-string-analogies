
import argparse
from collections import defaultdict
import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import checkpoint
import datasets as dat
from evaluate import evaluate_loss, evaluate_predictions
from model import MLC
from timing import timeSince

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(batch: defaultdict, model: MLC, loss_fn, optimizer) -> float:
    """Train MLC model on a batch of letter-string problems.

    Args:
        batch (defaultdict): A dictionary that represents a batch of letter-string problems.
        model (MLC): MLC seq2seq encoder-decoder.
        loss_fn (_type_): loss function (typically torch.nn.CrossEntropyLoss)
        optimizer (_type_): optimizer (typically torch.optim.adamw)

    Returns:
        float: The mean loss of the model predictions for this batch.
    """
    optimizer.zero_grad()
    model.train()
    # forward pass through MLC model
    if DEVICE == "cuda":
        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
            decoder_output = model(batch['yq_sos_padded'], batch) # returns (batch_size, max_target_length + 1, n_symbols)
    else:
        decoder_output = model(batch['yq_sos_padded'], batch) # returns (batch_size, max_target_length + 1, n_symbols)
    # flatten first two dimensions to pass to loss function:
    logits_flat = decoder_output.reshape(-1, decoder_output.shape[-1]) # (batch_size * max_target_length + 1, output_size)
    loss = loss_fn(logits_flat, batch['yq_padded'].reshape(-1))
    assert(not torch.isinf(loss))
    assert(not torch.isnan(loss))
    # backpropagate gradients and take optimization step:
    loss.backward()
    optimizer.step()
    return loss.cpu().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename_model', type=str, default='test.pt', help='*REQUIRED* Filename for saving model checkpoints. Ends in .pt')
    parser.add_argument('--dir_model', type=str, default='models', help='Directory for saving model files')
    parser.add_argument('--dir_data', type=str, default='data/base_problems', help='Directory for loading datasets')
    parser.add_argument('--batch_size', type=int, default=25, help='number of episodes per batch')
    parser.add_argument('--nepochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_end_factor', type=int, default=0.05, help='factor X for decrease learning rate linearly from 1.0*lr to X*lr across training')
    parser.add_argument('--lr_warmup', default=True, action='store_true', help='Turn off learning rate warm up (by default, we use 1 epoch of warm up)')
    parser.add_argument('--nlayers_encoder', type=int, default=3, help='number of layers for encoder')
    parser.add_argument('--nlayers_decoder', type=int, default=3, help='number of layers for decoder')
    parser.add_argument('--nheads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--emb_size', type=int, default=128, help='size of embedding')
    parser.add_argument('--ff_mult', type=int, default=4, help='multiplier for size of the fully-connected layer in transformer')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout applied to embeddings and transformer')        
    parser.add_argument('--act', type=str, default='gelu', help='activation function in the fully-connected layer of the transformer (relu or gelu)')
    parser.add_argument('--save_best', default=True, action='store_true', help='Save the "best model" according to validation loss.')
    parser.add_argument('--save_best_skip', type=float, default=0.2, help='Do not bother saving the "best model" for this fraction of early training')
    parser.add_argument('--resume', default=False, action='store_true', help='Resume training from a previous checkpoint')

    args = parser.parse_args()
    model_save_path = f"{args.dir_model}/{args.filename_model}"
    
    if args.resume:
        print("Attempting to resume training...")
        cp = checkpoint.CheckPoint(path=model_save_path, device=DEVICE)
        # load datasets
        train_dataloader, val_dataloader = cp.load_dataloaders(data_dir=args.dir_data, val_batch_size=5000)
        D_train = train_dataloader.dataset
        D_val = val_dataloader.dataset
        # setup loss function
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=D_train.langs['output'].PAD_idx)
        # load model, optimizer, scheduler
        model, optimizer, scheduler_epoch, epoch_start, step = cp.resume_training(args)
        print(f"Successfully loaded prerequisites.\nResuming training at epoch {epoch_start}.")
        # set training loop variables that have been recorded previously
        best_val_loss = cp.checkpoint["best_val_loss"]
        counter = 0 # num updates since the loss was last reported
        train_tracker = cp.checkpoint["train_tracker"]
        val_accuracy_by_epoch = cp.checkpoint["val_accuracy"]
    # training a new model
    else: 
        # initialize datasets and dataloaders:
        D_train = dat.LetterStringDataset(data_dir=args.dir_data, mode="train")
        train_dataloader = DataLoader(D_train, batch_size=args.batch_size, collate_fn=D_train.collate_fn, shuffle=True)
        D_val = dat.LetterStringDataset(data_dir=args.dir_data, mode="val")
        val_dataloader = DataLoader(D_val, batch_size=5000, collate_fn=D_val.collate_fn)
        print(f"Using datasets from directory {args.dir_data}")

        # setup model:
        model = MLC(
            hidden_size=args.emb_size, 
            input_size=D_train.langs['input'].n_symbols, 
            output_size=D_train.langs['output'].n_symbols,
            PAD_idx_input=D_train.langs['input'].PAD_idx, 
            PAD_idx_output=D_train.langs['output'].PAD_idx,
            nlayers_encoder=args.nlayers_encoder, 
            nlayers_decoder=args.nlayers_decoder,
            nhead=args.nheads,
            dropout_p=args.dropout,
            activation= args.act,
            ff_mult=args.ff_mult
        )
        model = model.to(device=DEVICE)
        print(model)

        # setup loss function and optimizer:
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=D_train.langs['output'].PAD_idx)
        optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr, betas=(0.9,0.95), weight_decay=0.01)
        if args.lr_warmup:
            print('    with LR warmup ON (1st epoch)')
            scheduler_epoch = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=args.lr_end_factor, total_iters=args.nepochs-2)
            nstep_epoch_estimate = math.floor(len(D_train)/args.batch_size)
            scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-4, end_factor=1.0, total_iters=nstep_epoch_estimate)
        else:            
            print('    with LR warmup OFF')
            scheduler_epoch = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=args.lr_end_factor, total_iters=args.nepochs-1)

        best_val_loss = float('inf')
        counter = 0 # num updates since the loss was last reported
        step = 0
        train_tracker = []
        val_accuracy_by_epoch = []
        epoch_start = 1

    nsteps_estimate = math.ceil(args.nepochs*len(D_train)/args.batch_size)
    sum_train_loss = 0.
    # group args in dict for checkpoint saving:
    params_state = {'langs': D_train.langs, 'emb_size':args.emb_size, 'input_size':D_train.langs['input'].n_symbols, 'output_size':D_train.langs['output'].n_symbols,
                    'dropout':args.dropout, 'nlayers_encoder':args.nlayers_encoder, 'nlayers_decoder':args.nlayers_decoder,
                    'nepochs':args.nepochs, 'batch_size':args.batch_size, 'activation':"gelu", 'args':args}

    print(f"Training on {DEVICE}.")
    start = time.time()
    
    # training loop:
    for epoch in range(epoch_start,args.nepochs+1):
        print("Epoch",epoch,"\n---------------------------------------------------------------")

        for train_batch in train_dataloader:
            loss = train(train_batch, model, loss_fn, optimizer)
            sum_train_loss += loss
            counter += 1
            step += 1  
                        
            if step in [1,25] or step % 100 == 0:
                mylr = optimizer.param_groups[0]['lr']
                avg_train_loss = sum_train_loss / counter
                # compute validation loss
                val_loss = evaluate_loss(val_dataloader, model, loss_fn=loss_fn)
                mytracker = {'epoch':epoch, 'step':step, 'lr':mylr, 'avg_train_loss':avg_train_loss, 'val_loss': val_loss}
                train_tracker.append(mytracker)
                prop_finished = step / nsteps_estimate
                print(f"{timeSince(start, prop_finished)}, Step: {step}, LR: {mylr:.7f}, TrainLoss: {avg_train_loss:.4f}, ValLoss: {val_loss:.4f}")
                # update best validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # save new best model if specified and epoch is past save_best_skip:
                    if args.save_best and (epoch > args.nepochs * args.save_best_skip):
                        checkpoint.save(model_save_path, step, epoch, model, optimizer, scheduler_epoch, train_tracker, val_accuracy_by_epoch, best_val_loss, params_state, is_best=True)
                # reset training averages
                sum_train_loss = 0.
                counter = 0
            # if learning rate warm-up, increase learning rate for each step within the first epoch
            if args.lr_warmup and epoch==1: 
                scheduler_warmup.step() 
        # after each epoch, calculate and save val accuracy:
        scores, trans_types, distribution = evaluate_predictions(val_dataloader, model, max_length=val_dataloader.dataset.yq_max+5, eval_type="max")
        val_accuracy = dict()
        val_accuracy["epoch"] = epoch
        val_accuracy["overall"] = np.mean(scores)
        for trans in np.unique(trans_types):
            val_accuracy[trans] = np.mean(scores[trans_types==trans])
        print(f"Val. Accuracy (after epoch {val_accuracy["epoch"]}):", end=" ")
        for dist in np.unique(distribution):
                val_accuracy[dist] = np.mean(scores[distribution==dist])
                print(f"{dist}-distribution: {val_accuracy[dist]:.3f},", end=" ")
        print("\n")
        val_accuracy_by_epoch.append(val_accuracy)
        
        # after each epoch, adjust the general learning rate
        if epoch>1 or not args.lr_warmup: 
            scheduler_epoch.step()
        checkpoint.save(model_save_path, step, epoch, model, optimizer, scheduler_epoch, train_tracker, val_accuracy_by_epoch, best_val_loss,params_state)
    print('Training complete.')


if __name__ == "__main__":
    main()
