import math
import functools
from collections import OrderedDict

from nnsight import LanguageModel
import torch.nn.functional as F
import numpy as np
import torch


class ActivationStorage:
    def __init__(self, batch, encoder, decoder, output_act) -> None:
        self.batch = batch
        self.encoder = encoder  # Dict: {'res': {...}, 'attn': {...}}
        self.decoder = decoder  # Dict: {'res': {...}, 'self_attn': {...}, 'cross_attn': {...}}
        self.output = output_act # The final model prediction

class TransformerIntervenor:
    def __init__(self, model):
        self.model = model
        self._original_forwards = OrderedDict()
        self._force_attn_return()

        self.nnsight_model = LanguageModel(model, device_map="cpu")
        # Dynamically find the encoder layers
        self.layers = self.nnsight_model.transformer.encoder.layers
        self.num_layers = len(self.layers)
        self.stored_activations = {}

    def _force_attn_return(self):
        # 1. Overwrite Self-Attention forward in Encoder
        for layer in self.model.transformer.encoder.layers:
            self._patch_attn(layer.self_attn)

        # 2. Overwrite Cross-Attention forward in Decoder
        for layer in self.model.transformer.decoder.layers:
            # We target multihead_attn which is the Cross-Attention mechanism
            self._patch_attn(layer.multihead_attn)
            self._patch_attn(layer.self_attn)

    def _patch_attn(self, attn_module):
        """Helper to patch attention modules, so that they return headwise attention."""
        self._original_forwards[attn_module] = attn_module.forward

        def new_forward(module, *args, **kwargs):
            kwargs['need_weights'] = True
            kwargs['average_attn_weights'] = False

            orig_fn = self._original_forwards[module]
            attn_output, attn_weights = orig_fn(*args, **kwargs)
            
            return attn_output, attn_weights

        attn_module.forward = functools.partial(new_forward, attn_module)

    def format_input_for_nnsight(self, batch):
        input = {'z_padded': batch['yq_io_padded'], 'batch': batch}
        return input

    def extract(self, batch, apply_intervention=False, intervention_layer=None, intervention_head=None):
        self.model.eval()
        # Local containers for proxies
        enc_res, enc_attn = {}, {}
        dec_res, dec_self_attn, dec_cross_attn = {}, {}, {}
        
        input_data = self.format_input_for_nnsight(batch)

        with self.nnsight_model.trace() as tracer:
            with tracer.invoke(input_data):
                # Initial embedding state
                enc_res['input'] = self.nnsight_model.transformer.encoder.layers[0].input.save()
                
                # encoder
                for i, layer in enumerate(self.nnsight_model.transformer.encoder.layers):
                    if apply_intervention:
                        if not intervention_layer or i == intervention_layer:
                            inter_attn_weights = self.stored_activations["intervention"].encoder["attn"][f"layer_{i}"]
                            # print(layer.self_attn.source) # michael's advice
                            self.interchange_intervention(
                                mha_module=layer.self_attn, 
                                interchange_attn_weights=inter_attn_weights,
                                head=intervention_head
                            )
                    enc_attn[f'layer_{i}'] = layer.self_attn.output[1].save()
                    enc_res[f'layer_{i}'] = layer.output.save()

                # decoder
                for i, layer in enumerate(self.nnsight_model.transformer.decoder.layers):
                    # Self-Attention (Masked)
                    dec_self_attn[f'layer_{i}'] = layer.self_attn.output[1].save()
                    # Cross-Attention (Encoder-Decoder)
                    dec_cross_attn[f'layer_{i}'] = layer.multihead_attn.output[1].save()
                    # Residual Output
                    dec_res[f'layer_{i}'] = layer.output.save()

                # --- FINAL OUTPUT ---
                # Assuming the model has a final output layer (e.g., self.nnsight_model.out)
                final_output = self.nnsight_model.output.save() 

        # Package into clean nested dictionaries after tracer exits
        encoder_data = {'res': {k: v.detach() for k, v in enc_res.items()}, 
                        'attn': {k: v.detach() for k, v in enc_attn.items()}}
        
        decoder_data = {'res': {k: v.detach() for k, v in dec_res.items()},
                        'self_attn': {k: v.detach() for k, v in dec_self_attn.items()},
                        'cross_attn': {k: v.detach() for k, v in dec_cross_attn.items()}}

        return ActivationStorage(batch, encoder_data, decoder_data, final_output)
    
    def run_and_save(self, batch, label="default", intervention=False, intervention_layer=None, intervention_head=None):
        """Run model on a batch and extract/save hidden states. Optionally applies intervention."""
        storage = self.extract(
            batch, 
            apply_intervention=intervention, 
            intervention_layer=intervention_layer, 
            intervention_head=intervention_head
        )
        self.stored_activations[label] = storage

    def decoder_lens(self, layer:int):
        ...

    def perturb_hidden_states(self, ):
        """Replace outputs of a given layer with another tensor in the forward pass."""
        ...

    @staticmethod
    def interchange_intervention(
        mha_module, 
        value_vectors = None,
        interchange_attn_weights = None, 
        head: int = None) -> None:
        """Insert attention pattern and replace it in the forward pass.

        Args:
            mha_module (_type_): MultiHeadAttention module
            value_vectors (Tensor, optional): Value vectors to replace normal value vectors when computing attn_weights @ v. Defaults to None.
            interchange_attn_weights (Tensor, optional): _description_. Defaults to None.
            head (int, optional): Only replace the original attention weights at `head=head`. Defaults to None, in which case all attn weights are replaced.

        """
        # retrieve dimensions
        x = mha_module.input # must be of shape (batch_size, seq_len, embed_dim)
        bsz, seq_len, embed_dim = x.shape
        num_heads = mha_module.num_heads
        head_dim = embed_dim // num_heads

        # Replicate In-Projection
        # This matches PyTorch internal _in_projection_packed
        combined_proj = F.linear(x, mha_module.in_proj_weight, mha_module.in_proj_bias)
        q, k, v = combined_proj.chunk(3, dim=-1) # Each is [1, 39, 128]

        # Reshape (bsz, seq_len, embed_dim) -> (bsz, num_heads, seq_len, head_dim)
        def reshape_heads(t):
            return t.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        # Scale Q by 1/sqrt(head_dim)
        q = q * (1.0 / math.sqrt(float(head_dim)))
        
        # orig_attn_weights: (bsz, num_heads, seq_len, seq_len)
        orig_attn_weights = q @ k.transpose(-2, -1)
        
        # Apply Softmax to match original attention behavior
        orig_attn_weights = F.softmax(orig_attn_weights, dim=-1)

        # Apply attention intervention if not None
        if interchange_attn_weights is not None:
            if head is not None:
                final_attn_weights = orig_attn_weights.clone()
                # Swap only the target head
                final_attn_weights[:, head, :, :] = interchange_attn_weights[:, head, :, :]
            else:
                final_attn_weights = interchange_attn_weights.view(bsz, num_heads, seq_len, seq_len)
        else:
            final_attn_weights = orig_attn_weights
        
        # attn weights-values matmul
        attn_output = final_attn_weights @ v

        # project back to (bsz, seq_len, embed_dim):
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, embed_dim)
        
        # Final linear projection
        attn_output = mha_module.out_proj(attn_output)

        mha_module.output = attn_output, final_attn_weights