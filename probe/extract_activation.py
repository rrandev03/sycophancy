import torch
from collections import defaultdict
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def extract_mha_activation(model, processor, inputs):
    inputs = inputs.to(model.device)
    
    if 'gemma' in str(type(model)).lower():
        NUM_LAYERS = model.config.text_config.num_hidden_layers
        HIDDEN_DIM = model.config.text_config.hidden_size
        NUM_HEADS = model.config.text_config.num_attention_heads
        HEAD_DIM = model.config.text_config.head_dim
    else:
        NUM_LAYERS = model.config.num_hidden_layers
        HIDDEN_DIM = model.config.hidden_size
        NUM_HEADS = model.config.num_attention_heads
        HEAD_DIM = model.config.head_dim

    hidden_states = defaultdict(list)

    for name, module in model.named_modules():
        module._forward_hooks.clear()
        module._backward_hooks.clear()
        if hasattr(module, '_forward_pre_hooks'):
            module._forward_pre_hooks.clear()

    def get_activation(name):
        def hook(module, input, output):
            # Store the output of each audio encoder layer
            hidden_states[name].append(input)
        return hook

    for handle in list(model._forward_hooks.values()):
        handle.remove()

    hooks = []
    for i in range(NUM_LAYERS):
        if 'gemma' in str(type(model)).lower():
            layer = model.model.layers[i].self_attn.o_proj # changed language_model to model
        else:
            layer = model.model.layers[i].self_attn.o_proj
        hook = layer.register_forward_hook(get_activation(f"mha_layer_{i}"))
        hooks.append(hook)

    # Your existing code for loading and processing the audio
    model.eval()

    # Just run forward pass instead of generate to avoid multiple hook calls
    with torch.no_grad():
        model_output = model(input_ids=inputs.unsqueeze(0))

    stacked_tensors = []
    for layer_name, states in hidden_states.items():
        if 'gemma' in str(type(model)).lower():
            layer_hidden_state = states[0][0][:, -1, :]  # output shape: [1, hidden_dim],  -3 since we take the last content token (not newline in -1 and eos token in -2)
            # last_index = len(inputs) - 1 - inputs.tolist()[::-1].index(105)
            # layer_hidden_state = states[0][0][:, last_index+3:-2, :]
            # layer_hidden_state = layer_hidden_state.mean(dim=1)  # shape: [1, hidden_dim]
        else:
            layer_hidden_state = states[0][0][:, -1, :]
            # last_index = len(inputs) - 1 - inputs.tolist()[::-1].index(128007)
            # layer_hidden_state = states[0][0][:, last_index+1:-1, :]
            # layer_hidden_state = layer_hidden_state.mean(dim=1)  # shape: [1, hidden_dim]
        new_shape = layer_hidden_state.size()[:-1] + (NUM_HEADS, HEAD_DIM) # shape: [1, num_head, head_dim]
        layer_hidden_state = layer_hidden_state.view(new_shape)
        stacked_tensors.append(layer_hidden_state)
    
    final_stacked_tensor = torch.cat(stacked_tensors, dim=0)
    return final_stacked_tensor

def extract_mlp_activation(model, processor, inputs):
    inputs = inputs.to(model.device)
    
    if 'gemma' in str(type(model)).lower():
        NUM_LAYERS = model.config.text_config.num_hidden_layers
        HIDDEN_DIM = model.config.text_config.hidden_size
        NUM_HEADS = model.config.text_config.num_attention_heads
        HEAD_DIM = model.config.text_config.head_dim
    else:
        NUM_LAYERS = model.config.num_hidden_layers
        HIDDEN_DIM = model.config.hidden_size
        NUM_HEADS = model.config.num_attention_heads
        HEAD_DIM = model.config.head_dim

    hidden_states = defaultdict(list)
    # Clear any existing data in the dictionary
    for name, module in model.named_modules():
        module._forward_hooks.clear()
        module._backward_hooks.clear()
        if hasattr(module, '_forward_pre_hooks'):
            module._forward_pre_hooks.clear()

    # Define hooks for the audio encoder layers
    def get_activation(name):
        def hook(module, input, output):
            hidden_states[name].append(input)
        return hook

    # Clear any existing hooks if you're running this multiple times
    for handle in list(model._forward_hooks.values()):
        handle.remove()

    # Register hooks to all audio encoder layers
    hooks = []
    for i in range(NUM_LAYERS):
        if 'gemma' in str(type(model)).lower():
            layer = model.language_model.model.layers[i].mlp.down_proj
        else:
            layer = model.model.layers[i].mlp.down_proj
        hook = layer.register_forward_hook(get_activation(f"mlp_layer_{i}"))
        hooks.append(hook)

    model.eval()

    # Just run forward pass instead of generate to avoid multiple hook calls
    with torch.no_grad():
        model_output = model(input_ids=inputs.unsqueeze(0))

    stacked_tensors = []
    for layer_name, states in hidden_states.items():
        if 'gemma' in str(type(model)).lower():
            # layer_hidden_state = states[0][0][:, -2, :]
            last_index = len(inputs) - 1 - inputs.tolist()[::-1].index(105)
            layer_hidden_state = states[0][0][:, last_index+3:-2, :]
            layer_hidden_state = layer_hidden_state.mean(dim=1)  # shape: [1, hidden_dim]
        else:
            # layer_hidden_state = states[0][0][:, -1, :] 
            last_index = len(inputs) - 1 - inputs.tolist()[::-1].index(128007)
            layer_hidden_state = states[0][0][:, last_index+1:-1, :]
            layer_hidden_state = layer_hidden_state.mean(dim=1)  # shape: [1, hidden_dim]
        stacked_tensors.append(layer_hidden_state)
    
    final_stacked_tensor = torch.cat(stacked_tensors, dim=0)
    return final_stacked_tensor

def extract_residual_activation(model, processor, inputs):
    inputs = inputs.to(model.device)
    
    if 'gemma' in str(type(model)).lower():
        NUM_LAYERS = model.config.text_config.num_hidden_layers
        HIDDEN_DIM = model.config.text_config.hidden_size
        NUM_HEADS = model.config.text_config.num_attention_heads
        HEAD_DIM = model.config.text_config.head_dim
    else:
        NUM_LAYERS = model.config.num_hidden_layers
        HIDDEN_DIM = model.config.hidden_size
        NUM_HEADS = model.config.num_attention_heads
        HEAD_DIM = model.config.head_dim

    hidden_states = defaultdict(list)
    # Clear any existing data in the dictionary
    for name, module in model.named_modules():
        module._forward_hooks.clear()
        module._backward_hooks.clear()
        if hasattr(module, '_forward_pre_hooks'):
            module._forward_pre_hooks.clear()

    # Define hooks for the audio encoder layers
    def get_activation(name):
        def hook(module, input, output):
            hidden_states[name].append(output)
        return hook

    # Clear any existing hooks if you're running this multiple times
    for handle in list(model._forward_hooks.values()):
        handle.remove()

    # Register hooks to all audio encoder layers
    hooks = []
    for i in range(NUM_LAYERS):
        if 'gemma' in str(type(model)).lower():
            layer = model.language_model.model.layers[i]
        else:
            layer = model.model.layers[i]
        hook = layer.register_forward_hook(get_activation(f"residual_layer_{i}"))
        hooks.append(hook)

    model.eval()

    # Just run forward pass instead of generate to avoid multiple hook calls
    with torch.no_grad():
        model_output = model(input_ids=inputs.unsqueeze(0))

    stacked_tensors = []
    for layer_name, states in hidden_states.items():
        if 'gemma' in str(type(model)).lower():
            # layer_hidden_state = states[0][0][:, -2, :]  # shape: [1, hidden_dim],  -3 since we take the last content token (not special tokens)
            last_index = len(inputs) - 1 - inputs.tolist()[::-1].index(105)
            layer_hidden_state = states[0][0][:, last_index+3:-2, :]
            layer_hidden_state = layer_hidden_state.mean(dim=1)  # shape: [1, hidden_dim]
        else:
            # layer_hidden_state = states[0][0][:, -1, :]  # output shape: [1, hidden_dim]
            last_index = len(inputs) - 1 - inputs.tolist()[::-1].index(128007)
            layer_hidden_state = states[0][0][:, last_index+1:-1, :]
            layer_hidden_state = layer_hidden_state.mean(dim=1)  # shape: [1, hidden_dim]
        stacked_tensors.append(layer_hidden_state)
    
    final_stacked_tensor = torch.cat(stacked_tensors, dim=0)
    return final_stacked_tensor