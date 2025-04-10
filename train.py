import datasets as dat
import torch
from evaluate import predict, evaluate_ll
from torch.utils.data import DataLoader
from model import MLC, describe_model
import torch.optim as optim
import math
import time
import argparse
from train_lib import timeSince

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(batch, net, loss_fn, optimizer):
    # Update the model for one batch (which is a set of episodes)
    #
    # Input
    #   batch : dict output from dat.make_biml_batch
    #   net : BIML model
    #   loss_fn : loss function
    #   optimizer : torch optimizer (AdamW)
    optimizer.zero_grad()
    net.train()
    m = len(batch['yq']) # effective batch size b*nq (num_episodes*num_queries)
    target_batches = batch['yq_padded'] # b*nq x max_length
    target_lengths = batch['yq_lengths'] # list of size b*nq
    target_shift = batch['yq_sos_padded'] # b*nq x max_length
        # shifted targets with padding (added SOS symbol at beginning and removed EOS symbol) 
    decoder_output = net(target_shift, batch) # b*nq x max_length x output_size
    logits_flat = decoder_output.reshape(-1, decoder_output.shape[-1]) # (b*nq*max_length, output_size)
    loss = loss_fn(logits_flat, target_batches.reshape(-1))
    assert(not torch.isinf(loss))
    assert(not torch.isnan(loss))
    loss.backward()
    optimizer.step()
    dict_loss = {}
    dict_loss['total'] = loss.cpu().item()
    return dict_loss


def save_checkpoint(fn_out_model, step, epoch, net, optimizer, scheduler_epoch, train_tracker, best_val_loss, params, is_best=False):
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
             'best_val_loss' : best_val_loss}
    state.update(params)
    torch.save(state, fn_out_model)
    print(' < Done. >')


def load_checkpoint(fn_out_model, net, optimizer, scheduler_epoch, params):
    # Note that the command line args must be the same now as when model was saved.
    #  (Except for 'resume' parameter)
    #
    # Input
    #  fn_out_model : filename for model to resume
    # ...
    #  params : list of hyperpameters
    # 
    # Output
    #  step : number of gradient steps at which we resume
    #  epoch_resume_start : number of completed epochs + 1
    #  train_tracker : array that stores losses over training
    #  best_val_loss : best validation loss so far (if using --save_best)
    checkpoint = torch.load(fn_out_model, map_location=DEVICE)
    
    net = net.to(device=DEVICE)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler_epoch.load_state_dict(checkpoint['scheduler_epoch_state_dict'])
    step = checkpoint['step']
    epoch = checkpoint['epoch']
    epoch_resume_start = epoch+1
    train_tracker = checkpoint['train_tracker']
    best_val_loss = checkpoint['best_val_loss']
    return step, epoch_resume_start, train_tracker, best_val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fn_out_model', type=str, default='', help='*REQUIRED* Filename for saving model checkpoints. Typically ends in .pt')
    parser.add_argument('--dir_model', type=str, default='out_models', help='Directory for saving model files')
    parser.add_argument('--query_from_supp', default=False, help='Whether or not to sample queries from the support examples shown to the model.')
    parser.add_argument('--batch_size', type=int, default=25, help='number of episodes per batch')
    parser.add_argument('--nepochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_end_factor', type=int, default=0.05, help='factor X for decrease learning rate linearly from 1.0*lr to X*lr across training')
    parser.add_argument('--no_lr_warmup', default=False, action='store_true', help='Turn off learning rate warm up (by default, we use 1 epoch of warm up)')
    parser.add_argument('--nlayers_encoder', type=int, default=3, help='number of layers for encoder')
    parser.add_argument('--nlayers_decoder', type=int, default=3, help='number of layers for decoder')
    parser.add_argument('--emb_size', type=int, default=128, help='size of embedding')
    parser.add_argument('--ff_mult', type=int, default=4, help='multiplier for size of the fully-connected layer in transformer')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout applied to embeddings and transformer')        
    parser.add_argument('--act', type=str, default='gelu', help='activation function in the fully-connected layer of the transformer (relu or gelu)')
    parser.add_argument('--save_best', default=False, action='store_true', help='Save the "best model" according to validation loss.')
    parser.add_argument('--save_best_skip', type=float, default=0.2, help='Do not bother saving the "best model" for this fraction of early training')
    parser.add_argument('--resume', default=False, action='store_true', help='Resume training from a previous checkpoint')

    args = parser.parse_args()
    # set hyperparameters:
    emb_size = 64
    nlayers_encoder = 3
    nlayers_decoder = 3
    p_dropout = 0.1
    lr = 0.001
    lr_end_factor=0.05
    lr_warmup = False
    batch_size = 1
    epoch_start = 0
    nepochs = 1
    model_path = "MLC_models/model.pt"


    # get datasets and dataloader:
    D_train = dat.LetterStringDataset(data_dir="data", mode="train")
    train_dataloader = DataLoader(D_train,batch_size=5,collate_fn=lambda x:dat.get_mlc_batch(x,D_train.langs),
                                    shuffle=True)
    D_val = dat.LetterStringDataset(data_dir="data", mode="val")
    val_dataloader = DataLoader(D_val,batch_size=5,collate_fn=lambda x:dat.get_mlc_batch(x,D_train.langs),
                                    shuffle=True)

    
    input_size = D_train.langs['input'].n_symbols, 
    output_size = D_train.langs['output'].n_symbols,
    # setup model:
    net = MLC(
        hidden_size=emb_size, 
        input_size=D_train.langs['input'].n_symbols, 
        output_size=D_train.langs['output'].n_symbols,
        PAD_idx_input=D_train.langs['input'].PAD_idx, 
        PAD_idx_output=D_train.langs['output'].PAD_idx,
        nlayers_encoder=nlayers_encoder, 
        nlayers_decoder=nlayers_decoder
        )
    net = net.to(device=DEVICE)
    describe_model(net)

    # group args in dict:
    params_state = {'langs': D_train.langs, 'emb_size':emb_size, 'input_size':input_size, 'output_size':output_size,
                    'dropout':p_dropout, 'nlayers_encoder':nlayers_encoder, 'nlayers_decoder':nlayers_decoder,
                    'nepochs':nepochs, 'batch_size':batch_size, 'activation':"gelu", 'args':args}
    # setup loss function and optimizer:
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=D_train.langs['output'].PAD_idx)
    optimizer = optim.AdamW(net.parameters(),lr=lr_end_factor, betas=(0.9,0.95), weight_decay=0.01)
    if lr_warmup:
        print('    with LR warmup ON (1st epoch)')
        scheduler_epoch = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=lr_end_factor, total_iters=nepochs-2, verbose=False)
        nstep_epoch_estimate = math.floor(len(D_train)/batch_size)
        scheduler_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-4, end_factor=1.0, total_iters=nstep_epoch_estimate, verbose=False)
    else:            
        print('    with LR warmup OFF')
        scheduler_epoch = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=lr_end_factor, total_iters=nepochs-1, verbose=False)

    nsteps_estimate = math.ceil(nepochs*len(D_train)/batch_size)
    avg_train_loss = 0.
    best_val_loss = float('inf')
    counter = 0 # num updates since the loss was last reported
    step = 0
    train_tracker = []
    val_accs = []
    epoch_start = 1
    start = time.time()
    # training loop:
    for epoch in range(epoch_start,nepochs+1):
        print("Epoch",epoch,"\n-------------------------------")

        for batch_idx, train_batch in enumerate(train_dataloader):
            train_batch = dat.set_batch_to_device(train_batch)
            dict_loss = train(train_batch, net, loss_fn, optimizer)
            print(f"Batch {batch_idx}, Loss:{dict_loss['total']:.4f}")
            avg_train_loss += dict_loss['total']
            counter += 1
            step += 1  
                         
            if step in [1,25] or step % 100 == 0:
                mylr = optimizer.param_groups[0]['lr']
                mytracker = {'epoch':epoch, 'step':step, 'lr':mylr, 'avg_train_loss':avg_train_loss/counter}
                print('{:s} ({:d} {:.0f}% finished) LR: {:.7f}, TrainLoss: {:.4f}, '.format(timeSince(start, float(step) / float(nsteps_estimate)),
                                         step, float(step) / float(nsteps_estimate) * 100., mylr, avg_train_loss/counter), end='')
                
                # compute val loss
                total_ll, total_N = evaluate_ll(val_dataloader, net, D_val.langs, loss_fn=loss_fn)
                val_loss = -total_ll/total_N
                print('ValLoss: {:.4f}'.format(val_loss))
                mytracker['val_loss'] = val_loss
                avg_train_loss = 0.
                counter = 0
                train_tracker.append(mytracker)

                if args.save_best and val_loss < best_val_loss and (epoch > nepochs * args.save_best_skip): # don't bother saving best model in early epochs
                    best_val_loss = val_loss
                    save_checkpoint(model_path,step,epoch,net,optimizer,scheduler_epoch,train_tracker,best_val_loss,params_state,is_best=True)

            # if warm-up, adjust learning rate for each step of the first epoch
            if lr_warmup and epoch==1: scheduler_warmup.step()
        
        # after each epoch, adjust the general learning rate
        if epoch>1 or not lr_warmup: scheduler_epoch.step()
        save_checkpoint(model_path,step,epoch,net,optimizer,scheduler_epoch,train_tracker,best_val_loss,params_state)

    print('Training complete.')
