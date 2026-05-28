"""Applies decoder to earlier layers of the encoder"""

import torch
from torch.distributions import Categorical

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DecoderLens:
    def __init__(self, model) -> None:
        """Applies the decoder lens to the outputs of an earlier attention layer of the MLC model.

        Args:
            model (MLC): _description_
            encoder_layer (int): _description_
        """
        self.model = model
    
    @torch.no_grad()
    def predict_batch(
        self,
        memory,
        batch, 
        langs, 
        max_length: int,
        check_for_valid_length: bool=True,
        eval_type='max', 
        return_logits: bool=False
        ):
        """Generate outputs using earlier encoder layers' output for cross attention. \
        Generates outputs for problem batch until max_length is reached or EOS is found.

        Args:
            memory: Memory from an (earlier) encoder block of the model.
            batch (_type_): generated from `torch.utils.Dataloader` with `datasets.LetterStringDataset.collate_fn()`
            model (model.MLC): MLC model
            langs (dict[datasets.Lang]): dict of datasets.Lang classes
            max_length (int): maximum length of generated output sequences
            eval_type (str, optional): 'max' for greedy decoding, 'sample' for sample from distribution. Default 'max'.

        Returns:
            list[list[str]]: A list of lists that contain the model predictions (symbols) for each problem in the batch.
        """
        assert eval_type in ['max','sample']
        if check_for_valid_length:
            max_length_target = batch['yq_padded'].shape[1]-1 # length without EOS
            assert max_length >= max_length_target # make sure that the model can generate targets of the proper length
        self.model.eval()
        emission_lang = langs['output']
        memory_padding_mask = batch['xq_context_padded'] == self.model.PAD_idx_input
        batch_size = len(batch['xq_context'])
        z_padded = torch.tensor([emission_lang.IN_OUT_idx]*batch_size, device=DEVICE)
        z_padded = z_padded.unsqueeze(1)

        # Run through decoder
        all_decoder_outputs = torch.zeros((batch_size, max_length), dtype=torch.long, device=DEVICE)
        all_logits = torch.zeros((batch_size, emission_lang.n_symbols, max_length), dtype=torch.float32, device=DEVICE)
        for t in range(max_length):
            decoder_output = self.model.decode(z_padded, memory, memory_padding_mask)
            decoder_output = decoder_output[:,-1] # get the last step's output (batch_size, output_size)
            all_logits[:,:,t] = decoder_output
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
        yq_predict = [emission_lang.tensor_to_symbols(all_decoder_outputs[i,:].view(-1)) for i in range(batch_size)]
        return (yq_predict, all_logits) if return_logits else yq_predict
