# Segment Anything Model 2 (SAM 2) - Memory Attention Study Notes

**Overview:**
Segment Anything Model 2 (SAM 2) is designed to handle promptable visual segmentation in images and videos. One of the core components of SAM 2 is the memory attention mechanism, which enhances the model's ability to process and segment sequences of frames in videos.

**Key Components from `sam2/modeling/memory_attention.py`:**

1. **MemoryAttentionLayer Class:**
   - **Purpose:** Implements a single layer of memory attention, which includes both self-attention and cross-attention mechanisms.
   - **Initialization Parameters:**
     - `activation`: Activation function (e.g., 'relu').
     - `cross_attention`: Module for cross-attention.
     - `d_model`: Dimensionality of the model.
     - `dim_feedforward`: Dimensionality of the feedforward network.
     - `dropout`: Dropout rate.
     - `pos_enc_at_attn`: Boolean indicating whether to add positional encoding at self-attention.
     - `pos_enc_at_cross_attn_keys`: Boolean indicating whether to add positional encoding at cross-attention keys.
     - `pos_enc_at_cross_attn_queries`: Boolean indicating whether to add positional encoding at cross-attention queries.
     - `self_attention`: Module for self-attention.
   - **Methods:**
     - `_forward_sa`: Handles self-attention by normalizing the target, adding positional encoding if specified, and applying self-attention followed by dropout.
     - `_forward_ca`: Handles cross-attention by normalizing the target, adding positional encoding if specified, and applying cross-attention followed by dropout.
     - `forward`: Integrates self-attention, cross-attention, and feedforward neural network, applying them sequentially to the target and memory inputs.

2. **MemoryAttention Class:**
   - **Purpose:** Comprises multiple layers of memory attention to create a deep model capable of handling complex sequences.
   - **Initialization Parameters:**
     - `d_model`: Dimensionality of the model.
     - `pos_enc_at_input`: Boolean indicating whether to add positional encoding at the input.
     - `layer`: A single layer of memory attention (`MemoryAttentionLayer`).
     - `num_layers`: Number of layers in the memory attention module.
     - `batch_first`: Boolean indicating whether the input tensors are batch-first.
   - **Methods:**
     - `forward`: Processes the input tensors through multiple layers of memory attention, applying self-attention and cross-attention in each layer and normalizing the output.

**Configuration Files:**
The configuration files define different model variants using the memory attention mechanism. Below are examples of the configuration settings:

- **sam2_configs/sam2_hiera_t.yaml**
  ```yaml
  memory_attention:
    _target_: sam2.modeling.memory_attention.MemoryAttention
    d_model: 256
    layer:
      _target_: sam2.modeling.memory_attention.MemoryAttentionLayer
      activation: relu
  ```

- **sam2_configs/sam2_hiera_l.yaml**
  ```yaml
  memory_attention:
    _target_: sam2.modeling.memory_attention.MemoryAttention
    d_model: 256
    layer:
      _target_: sam2.modeling.memory_attention.MemoryAttentionLayer
      activation: relu
  ```

- **sam2_configs/sam2_hiera_s.yaml**
  ```yaml
  memory_attention:
    _target_: sam2.modeling.memory_attention.MemoryAttention
    d_model: 256
    layer:
      _target_: sam2.modeling.memory_attention.MemoryAttentionLayer
      activation: relu
  ```

- **sam2_configs/sam2_hiera_b+.yaml**
  ```yaml
  memory_attention:
    _target_: sam2.modeling.memory_attention.MemoryAttention
    d_model: 256
    layer:
      _target_: sam2.modeling.memory_attention.MemoryAttentionLayer
      activation: relu
  ```

**Conclusion:**
The memory attention mechanism in SAM 2 is crucial for handling the temporal dependencies and spatial features in video segmentation tasks. By integrating self-attention and cross-attention within a multi-layered structure, SAM 2 effectively processes and segments sequences, making it highly versatile for various visual domains and tasks.

## Citing SAM 2

If you use SAM 2 or the SA-V dataset in your research, please use the following BibTeX entry.

```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint},
  year={2024}
}
```
