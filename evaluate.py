import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm

import datasets as dat

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict(batch, model, langs, max_length, eval_type='max', out_mask_allow=[]):
    # Predicts outputs for problem batch until max_length is reached or EOS is found.
    #
    #  Input
    #   batch : from dat.make_biml_batch
    #   model : MLC model
    #   max_length : maximum length of output sequences
    #   langs : dict of dat.Lang classes
    #   eval_type : 'max' for greedy decoding, 'sample' for sample from distribution
    #   out_mask_allow : default=[]; list of emission symbols (strings) we want to allow. Default of [] allows all output emissions
    assert eval_type in ['max','sample']
    model.eval()
    emission_lang = langs['output']
    use_mask = len(out_mask_allow)>0
    memory, memory_padding_mask = model.encode(batch) 
        # memory : b*nq x maxlength_src x hidden_size
        # memory_padding_mask : b*nq x maxlength_src (False means leave alone)
    m = len(batch['yq']) # b*nq
    z_padded = torch.tensor([emission_lang.symbol2index[dat.SOS_token]]*m, device=DEVICE) # b*nq length tensor
    z_padded = z_padded.unsqueeze(1) # [b*nq x 1] tensor
    # z_padded = z_padded.to(device=DEVICE)
    max_length_target = batch['yq_padded'].shape[1]-1 # length without EOS
    assert max_length >= max_length_target # make sure that the model can generate targets of the proper length

    # make the output mask if certain emissions are restricted
    if use_mask:
        assert dat.EOS_token in out_mask_allow # EOS must be included as an allowed symbol
        additive_out_mask = -torch.inf * torch.ones((m,model.output_size), dtype=torch.float, device=DEVICE)
        # additive_out_mask = additive_out_mask.to(device=DEVICE)
        for s in out_mask_allow:
            sidx = langs['output'].symbol2index[s]
            additive_out_mask[:,sidx] = 0.

    # Run through decoder
    all_decoder_outputs = torch.zeros((m, max_length), dtype=torch.long, device=DEVICE)
    # all_decoder_outputs = all_decoder_outputs.to(device=DEVICE)
    for t in range(max_length):
        decoder_output = model.decode(z_padded, memory, memory_padding_mask)
            # decoder_output is b*nq x (t+1) x output_size
        decoder_output = decoder_output[:,-1] # get the last step's output (batch_size x output_size)
        if use_mask: decoder_output += additive_out_mask

        # Choose the symbols at next timestep
        if eval_type == 'max': # pick the most likely
            topi = torch.argmax(decoder_output,dim=1)
            emissions = topi.view(-1)
        elif eval_type == 'sample':
            emissions = Categorical(logits=decoder_output).sample()
        all_decoder_outputs[:,t] = emissions
        z_padded = torch.cat([z_padded, emissions.unsqueeze(1)], dim=1)

    # Get predictions as strings and see if they are correct
    all_decoder_outputs = all_decoder_outputs.detach()
    yq_predict = [] # list of all predicted query outputs as strings
    for q in range(m):
        myseq = emission_lang.tensor_to_symbols(all_decoder_outputs[q,:].view(-1))
        yq_predict.append(myseq)
    
    return yq_predict


def evaluate_ll(dataloader: torch.utils.data.DataLoader, model, langs, loss_fn=[], p_lapse=0.0):
    # Evaluate the total (sum) log-likelihood across the entire validation set
    # 
    # Input
    #   dataloader: A torch.utils.DataLoader that represents validation/ test set.
    #   model : MLC model
    #   langs : dict of dat.Lang classes
    #   p_lapse : (default 0.) combine decoder outputs (prob 1-p_lapse) as mixture with uniform distribution (prob p_lapse)
    model.eval()
    total_N = 0
    total_ll = 0
    if not loss_fn: loss_fn = torch.nn.CrossEntropyLoss(ignore_index=langs['output'].PAD_idx)
    with torch.no_grad():
        for batch in dataloader:
            dict_loss = batch_ll(batch, model, loss_fn, langs, p_lapse=p_lapse)
            total_ll += dict_loss['ll']
            total_N += dict_loss['N']
    return total_ll, total_N


def batch_ll(batch, model, loss_fn, langs, p_lapse=0.0):
    # Evaluate log-likelihood (average over cells, and sum total) for a given batch
    #
    # Input
    #   batch : from dat.make_biml_batch
    #   loss_fn : loss function
    #   langs : dict of dat.Lang classes
    model.eval()
    target_batches = batch['yq_padded'] # b*nq x max_length
    target_lengths = batch['yq_lengths'] # list of size b*nq
    # Shifted targets with padding (added SOS symbol at beginning and removed EOS symbol) 
    target_shift = batch['yq_sos_padded'] # b*nq x max_length
    decoder_output = model(target_shift, batch) # b*nq x max_length x output_size    

    logits_flat = decoder_output.reshape(-1, decoder_output.shape[-1]) # (batch*max_len, output_size)
    if p_lapse > 0:
        logits_flat = smooth_decoder_outputs(logits_flat,p_lapse,langs['output'].symbols+[dat.EOS_token],langs)
    loss = loss_fn(logits_flat, target_batches.reshape(-1))
    loglike = -loss.cpu().item()
    dict_loss = {}
    dict_loss['ll_by_cell'] = loglike # average over cells
    dict_loss['N'] = float(sum(target_lengths)) # total number of valid cells
    dict_loss['ll'] = dict_loss['ll_by_cell'] * dict_loss['N'] # total LL
    return dict_loss


def evaluate_predictions(dataloader: torch.utils.data.DataLoader, model, max_length: int, eval_type='max') -> tuple:
    """Evaluates whether model predictions exactly match the solutions across entire data set.

    Args:
        dataloader (torch.DataLoader): dataloader build from datasets.LetterStringDataset
        model (model.MLC): MLC model
        max_length (int): maximum output length for each problem
        eval_type (str, optional): "max": take the maximum token of the logits, "sample" sample from the logit distribution. Defaults to 'max'.

    Returns:
        tuple: A tuple of a lists of scores (True/False) and transformation types (successor, sort, etc.) for each problem.
    """
    
    model.eval()
    scores = []
    transformation_types = []
    with torch.no_grad():
        for batch in tqdm(dataloader): # each batch
            predictions = predict(batch, model, dataloader.dataset.langs, max_length, eval_type=eval_type)
            batch_scores = [pred==yq for pred, yq in zip(predictions, batch["yq"])]
            scores += batch_scores
            transformation_types += batch["transformation"]

    return (np.array(scores), np.array(transformation_types))

def smooth_decoder_outputs(logits_flat,p_lapse,lapse_symb_include,langs):
    # Mix decoder outputs (logits_flat) with uniform distribution over allowed emissions (in lapse_symb_include)
    #
    # Input
    #  logits_flat : (batch*max_len, output_size) # unnomralized log-probabilities
    #  p_lapse : probability of a uniform lapse
    #  lapse_symb_include : list of tokens (strings) that we want to include in the lapse model
    #  langs : dict of dat.Lang classes
    #
    # Output
    #  log_probs_flat : (batch*max_len, output_size) normalized log-probabilities
    lapse_idx_include = [langs['output'].symbol2index[s] for s in lapse_symb_include]
    assert dat.SOS_token not in lapse_symb_include # SOS should not be an allowed output through lapse model
    sz = logits_flat.size() # get size (batch*max_len, output_size)
    probs_flat = F.softmax(logits_flat,dim=1) # (batch*max_len, output_size)
    num_classes_lapse = len(lapse_idx_include)
    probs_lapse = torch.zeros(sz, dtype=torch.float, device=DEVICE)
    # probs_lapse = probs_lapse.to(device=DEVICE)
    probs_lapse[:,lapse_idx_include] = 1./float(num_classes_lapse)
    log_probs_flat = torch.log((1-p_lapse)*probs_flat + p_lapse*probs_lapse) # (batch*max_len, output_size)
    return log_probs_flat
