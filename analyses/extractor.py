"""Extracts attention weights and hidden states from the MLC models' encoder."""

from collections import OrderedDict
import functools
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Extractor:
    def __init__(self, model):
        """Extracts hidden states and attention weights from both Encoder and Decoder."""
        self.model = model
        self.inputs = OrderedDict()
        self.hidden_states = OrderedDict()
        
        # Separate variables for attention to keep logic clean
        self.attention = OrderedDict()          # Stores Encoder Self-Attention
        self.decoder_attention = OrderedDict()  # Stores Decoder Cross-Attention
        
        self.hooks = []
        self._original_forwards = {}

    def _hook(self, name):
        def fn(module, inp, out):
            # Maintain original behavior for encoder inputs
            if name == "layer_0":
                inp_normed = self.model.transformer.encoder.norm(inp[0])
                self.inputs["encoder"] = inp_normed.detach().contiguous()
            
            if hasattr(out, "is_nested") and out.is_nested:
                out = out.to_padded_tensor(0.0)
            
            # Encoder applies norm after layers
            out = self.model.transformer.encoder.norm(out)
            self.hidden_states[name] = out.detach().contiguous()
        return fn

    def register(self):
        # 1. Register Encoder (Self-Attention)
        # Keeps original naming: 'layer_0', 'layer_1', etc.
        for i, layer in enumerate(self.model.transformer.encoder.layers):
            name = f"layer_{i}"
            h = layer.register_forward_hook(self._hook(name))
            self.hooks.append(h)
            
            self._patch_attn(layer.self_attn, name, self.attention)

        # 2. Register Decoder (Cross-Attention)
        # Uses 'layer_0', 'layer_1', etc. inside the decoder_attention dict
        for i, layer in enumerate(self.model.transformer.decoder.layers):
            name = f"layer_{i}"
            
            # We target multihead_attn which is the Cross-Attention mechanism
            self._patch_attn(layer.multihead_attn, name, self.decoder_attention)

    def _patch_attn(self, attn_module, layer_name, storage_dict):
        """Helper to patch attention modules without duplicating code."""
        self._original_forwards[attn_module] = attn_module.forward

        def new_forward(module, *args, **kwargs):
            kwargs['need_weights'] = True
            kwargs['average_attn_weights'] = False

            orig_fn = self._original_forwards[module]
            attn_output, attn_weights = orig_fn(*args, **kwargs)
            
            if attn_weights is not None:
                storage_dict[layer_name] = attn_weights.detach()
            
            return attn_output, attn_weights

        attn_module.forward = functools.partial(new_forward, attn_module)

    def clear(self):
        self.hidden_states.clear()
        self.attention.clear()
        self.decoder_attention.clear()
        self.inputs.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        for module, orig_forward in self._original_forwards.items():
            module.forward = orig_forward
        self._original_forwards.clear()


def check_outputs_unchanged_by_extractor(extractor, model, batch):
    """Test that the outputs of the model are not changed by the Extractor."""
    model.eval()
    with torch.no_grad():
        baseline_output = model(batch["yq_io_padded"], batch)

    # 2. Get extracted version
    extractor = Extractor(model)
    extractor.register()
    with torch.no_grad():
        extracted_output = model(batch["yq_io_padded"], batch)

    # 3. Compare
    diff = torch.abs(baseline_output - extracted_output).max().item()
    print(f"Max difference in logits: {diff}")

    # Use a standard tolerance for FP32 (usually 1e-6 or 1e-7)
    assert diff < 1e-6, "Extractor altered model behavior!"
