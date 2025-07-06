# Note

The original implementation of MeshCNN is too old. It has the following errors:
- Extremely Slow
- Incompatible with the latesest CUDA kernel
    - cause the matmul error

We may use the implementation from PyG