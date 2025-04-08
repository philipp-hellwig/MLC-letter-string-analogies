import torch
from torch.distributions import Categorical
import datasets as dat

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict(batch, net, langs, max_length, eval_type='max', out_mask_allow=[]):
    # Predicts outputs for problem batch until max_length is reached or EOS is found.
    #
    #  Input
    #   batch : from dat.make_biml_batch
    #   net : MLC model
    #   max_length : maximum length of output sequences
    #   langs : dict of dat.Lang classes
    #   eval_type : 'max' for greedy decoding, 'sample' for sample from distribution
    #   out_mask_allow : default=[]; list of emission symbols (strings) we want to allow. Default of [] allows all output emissions
    assert eval_type in ['max','sample']
    net.eval()
    emission_lang = langs['output']
    use_mask = len(out_mask_allow)>0
    memory, memory_padding_mask = net.encode(batch) 
        # memory : b*nq x maxlength_src x hidden_size
        # memory_padding_mask : b*nq x maxlength_src (False means leave alone)
    m = len(batch['yq']) # b*nq
    z_padded = torch.tensor([emission_lang.symbol2index[dat.SOS_token]]*m) # b*nq length tensor
    z_padded = z_padded.unsqueeze(1) # [b*nq x 1] tensor
    z_padded = z_padded.to(device=DEVICE)
    max_length_target = batch['yq_padded'].shape[1]-1 # length without EOS
    assert max_length >= max_length_target # make sure that the net can generate targets of the proper length

    # make the output mask if certain emissions are restricted
    if use_mask:
        assert dat.EOS_token in out_mask_allow # EOS must be included as an allowed symbol
        additive_out_mask = -torch.inf * torch.ones((m,net.output_size), dtype=torch.float)
        additive_out_mask = additive_out_mask.to(device=DEVICE)
        for s in out_mask_allow:
            sidx = langs['output'].symbol2index[s]
            additive_out_mask[:,sidx] = 0.

    # Run through decoder
    all_decoder_outputs = torch.zeros((m, max_length), dtype=torch.long)
    all_decoder_outputs = all_decoder_outputs.to(device=DEVICE)
    for t in range(max_length):
        decoder_output = net.decode(z_padded, memory, memory_padding_mask)
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