import torch

class Perturber:
    def __init__(self, model) -> None:
        self.model = model
        self.hidden_state_perturbation = None
        # new implementation:
        # self.hidden_state_masks = None
        # self.perturbed_states = None

    def insert_hidden_states(self, hidden_states: torch.Tensor, layer: int):
        """Insert hidden states that will be used in the next forward pass of the model.

        Args:
            hidden_states (torch.Tensor): hidden states to insert.
            layer (int): Layer in which to insert them
        """
        #TODO: change this to masks instead
        self.hidden_state_perturbation = {
            "layer": layer,
            "hidden_states": hidden_states
        }
    
    def insert_attention_pattern(self, pattern, layer: int, head: int):
        """Insert an attention pattern into a layer head of the encoder and use this pattern instead for the forward pass

        Args:
            pattern (_type_): _description_
            layer (int): _description_
            head (int): _description_
        """
        ...

    @torch.no_grad()
    def get_encoder_outputs(self, batch):
        """Generate encoder outputs with the applied perturbations."""
        # normal forward pass if 
        if not self.hidden_state_perturbation:
            hidden_states, _ = self.model.encode(batch)
            return hidden_states

        #TODO: store intermediate layer states and 
        # apply masks at each step to the hidden states in the encoder forward pass:
        start_layer_idx = self.hidden_state_perturbation["layer"] + 1
        hidden_states = self.hidden_state_perturbation["hidden_states"]
        
        # if perturbation is already at the final layer, return perturbed states directly:
        if start_layer_idx >= self.model.nlayers_encoder:
            return hidden_states
        else:
            # Get padding mask (required for Transformer layers to ignore PAD tokens)
            xq_context_padded = batch['xq_context_padded']
            src_padding_mask = xq_context_padded == self.model.PAD_idx_input
            # Iterate through remaining encoder layers:
            for i in range(start_layer_idx, self.model.nlayers_encoder):
                hidden_states = self.model.transformer.encoder.layers[i](
                    hidden_states, 
                    src_mask=None, 
                    src_key_padding_mask=src_padding_mask
                )
            return self.model.transformer.encoder.norm(hidden_states)
        
