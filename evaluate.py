import numpy as np
import torch
from torch.distributions import Categorical
import torch.utils.data
from tqdm import tqdm

import datasets

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@torch.no_grad()
def predict(batch, model, langs: datasets.Lang, max_length: int, eval_type='max') -> list:
    """Predicts outputs for problem batch until max_length is reached or EOS is found.

    Args:
        batch (_type_): generated from `torch.utils.Dataloader` with `datasets.LetterStringDataset.collate_fn()`
        model (model.MLC): MLC model
        langs (dict[datasets.Lang]): dict of datasets.Lang classes
        max_length (int): maximum length of generated output sequences
        eval_type (str, optional): 'max' for greedy decoding, 'sample' for sample from distribution. Default 'max'.

    Returns:
        list[list[str]]: A list of lists that contain the model predictions (symbols) for each problem in the batch.
    """
    assert eval_type in ['max','sample']
    max_length_target = batch['yq_padded'].shape[1]-1 # length without EOS
    assert max_length >= max_length_target # make sure that the model can generate targets of the proper length
    model.eval()
    emission_lang = langs['output']
    memory, memory_padding_mask = model.encode(batch) 
    batch_size = len(batch['yq'])
    z_padded = torch.tensor([emission_lang.symbol2index[datasets.SOS_token]]*batch_size, device=DEVICE)
    z_padded = z_padded.unsqueeze(1)

    # Run through decoder
    all_decoder_outputs = torch.zeros((m, max_length), dtype=torch.long, device=DEVICE)
    for t in range(max_length):
        decoder_output = model.decode(z_padded, memory, memory_padding_mask)
        decoder_output = decoder_output[:,-1] # get the last step's output (batch_size, output_size)

        # Choose the symbols at next timestep
        if eval_type == 'max': # pick the most likely
            top_id = torch.argmax(decoder_output,dim=1)
            emissions = top_id.view(-1)
        elif eval_type == 'sample':
            emissions = Categorical(logits=decoder_output).sample()
        all_decoder_outputs[:,t] = emissions
        z_padded = torch.cat([z_padded, emissions.unsqueeze(1)], dim=1)

    # Get predictions as strings and see if they are correct
    all_decoder_outputs = all_decoder_outputs.detach()
    yq_predict = [emission_lang.tensor_to_symbols(all_decoder_outputs[q,:].view(-1)) for q in range(m)]
    return yq_predict


def evaluate_loss(dataloader: torch.utils.data.DataLoader, model, loss_fn=None):
    """Compute average loss over (validation) dataset contained in the dataloader."""
    batch_losses = [batch_loss(batch, model, loss_fn) for batch in dataloader]
    return np.mean(batch_losses)


@torch.no_grad()
def batch_loss(batch, model, loss_fn):
    """Evaluate loss for a given batch"""
    model.eval()
    decoder_output = model(batch['yq_sos_padded'], batch) # b*nq x max_length x output_size    
    logits_flat = decoder_output.reshape(-1, decoder_output.shape[-1]) # (batch*max_len, output_size)
    loss = loss_fn(logits_flat, batch['yq_padded'].reshape(-1))
    return loss.cpu().item()


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
    scores = []
    transformation_types = []
    for batch in tqdm(dataloader, desc="Evaluating predicted solutions"): # each batch
        predictions = predict(batch, model, dataloader.dataset.langs, max_length, eval_type=eval_type)
        batch_scores = [pred==yq for pred, yq in zip(predictions, batch["yq"])]
        scores += batch_scores
        transformation_types += batch["transformation"]

    return np.array(scores), np.array(transformation_types)
