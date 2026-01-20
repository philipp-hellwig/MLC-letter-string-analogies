import os
import streamlit.components.v1 as components

_RELEASE = False  # Switch to True when you run 'npm build'

if not _RELEASE:
    # During development, both can point to the same dev server.
    # Your JS code will use the 'componentName' to decide what to render.
    _MAP_FUNC = components.declare_component("d3_attention_map", url="http://127.0.0.1:3001")
    _ENCODER_GRAPH_FUNC = components.declare_component("d3_computational_graph", url="http://127.0.0.1:3001")
else:
    # In production, they all serve from the same build folder
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend")
    _MAP_FUNC = components.declare_component("d3_attention_map", path=build_dir)
    _ENCODER_GRAPH_FUNC = components.declare_component("d3_computational_graph", path=build_dir)


def d3_attention_map(matrix, labels, title="Attention Heatmap", key=None):
    return _MAP_FUNC(
        matrix=matrix, 
        labels=labels, 
        title=title,
        view="d3_attention_map",
        key=key, 
        default=None
    )


def d3_computational_graph(attention_weights, hidden_states, labels, key=None, separators=[], prediction_data=[],
            decoder_layer_index=2, deleted_list=[]):
    """
    attention_weights: List of List of Lists (3D array: [layer][row][col])
    labels: List of strings (tokens)
    """
    return _ENCODER_GRAPH_FUNC(
        attention_weights=attention_weights, 
        hidden_states=hidden_states, 
        labels=labels, 
        view="full_graph", 
        key=key,
        separators=separators,
        prediction_data=prediction_data,
        decoder_layer_index=decoder_layer_index,
        deleted_list=deleted_list
    )
