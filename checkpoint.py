import torch
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import model
import datasets as dat


class CheckPoint:
    def __init__(self, path: str, device: torch.device):
        self.checkpoint = torch.load(path, map_location=device)
        self.train_data = pd.DataFrame(self.checkpoint["train_tracker"])
        self.device = device
    
    def plot_learning_rate(self, ax) -> plt.Axes:
        lr_plot = sns.lineplot(data=self.train_data, x="step", y="lr", ax=ax)
        return lr_plot

    def plot_loss(self, ax) -> plt.Axes:
        loss_data = pd.melt(self.train_data, id_vars=['step'], value_vars=['avg_train_loss','val_loss'], var_name="loss")
        loss_data["loss"] = loss_data["loss"].str.replace("_loss", "")
        loss_plot = sns.lineplot(data=loss_data, x='step', y='value', hue='loss', ax=ax)
        return loss_plot
    
    def load_model(self, verbose: bool=True) -> model.MLC:
        """Load MLC model using a checkpoint file (.pt).

        Args:
            verbose(bool): whether or not to print model and training information

        Returns:
            model.MLC
        """

        # Initialize model architecture:         
        net = model.MLC(
            hidden_size=self.checkpoint['emb_size'], 
            input_size=self.checkpoint['langs']['input'].n_symbols, 
            output_size=self.checkpoint['langs']['output'].n_symbols,
            PAD_idx_input=self.checkpoint["langs"]['input'].PAD_idx, 
            PAD_idx_output=self.checkpoint["langs"]['output'].PAD_idx,
            nlayers_encoder=self.checkpoint['nlayers_encoder'], 
            nlayers_decoder=self.checkpoint['nlayers_decoder'], 
            dropout_p=self.checkpoint['dropout'], 
            activation=self.checkpoint['activation']
        ) 
        # load trained parameters:    
        nets_state_dict = self.checkpoint['nets_state_dict']
        net.load_state_dict(nets_state_dict)
        net = net.to(device=self.device)

        if verbose:
            best_val_loss = -float('inf')
            if 'best_val_loss' in self.checkpoint: best_val_loss = self.checkpoint['best_val_loss']
            print('Loading model that has completed (or started) ' + str(self.checkpoint['epoch']) + ' of ' + str(self.checkpoint['nepochs']) + ' epochs')
            print('\tbatch size:', self.checkpoint['batch_size'])
            print(f"\tnumber of steps:{self.checkpoint['step']:,}")
            print('\tbest val loss achieved: {:.4f}'.format(best_val_loss))
            print(net)
        
        return net
    
    def load_dataloaders(self, data_dir="data", verbose: bool=True, val_batch_size=25):            
        # Load validation dataset
        D_train = dat.LetterStringDataset(data_dir=data_dir, mode="train")
        D_val = dat.LetterStringDataset(data_dir=data_dir, mode="val")
        train_dataloader = DataLoader(D_train,batch_size=self.checkpoint["batch_size"], collate_fn=D_train.collate_fn)
        val_dataloader = DataLoader(D_val,batch_size=val_batch_size, collate_fn=D_val.collate_fn)
        if verbose:
            print(f"Loading training ({len(D_train):,} samples) and validation ({len(D_val):,} samples) dataloaders.")
        return (train_dataloader, val_dataloader)


def save(fn_out_model, step, epoch, net, optimizer, scheduler_epoch, train_tracker, val_accuracies, best_val_loss, params, is_best=False):
    # Input
    #  fn_out_model : filename for saving the model
    #  step : number of gradient steps
    # ..
    #  train_tracker : array that stores losses over training
    #  best_val_loss : best validation loss so far (if using --save_best)
    #  params : list of hyperpameters
    #  is_best : special filename if best file so far  ... 'filename_best.pt'
    if is_best:
        s = fn_out_model.rsplit('.',1) # split off extension 
        fn_out_model = s[0] + '_best.' + s[1]
        print('> Saving new *best* model as',fn_out_model, end='')
    else:
        print('> Saving model as',fn_out_model, end='')
    state = {'step' : step,
            'epoch' : epoch,
            'nets_state_dict' : net.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'scheduler_epoch_state_dict' : scheduler_epoch.state_dict(),
            'train_tracker' : train_tracker,
            'val_accuracy' : val_accuracies,
            'best_val_loss' : best_val_loss}
    state.update(params)
    torch.save(state, fn_out_model)
    print(' < Done. >')