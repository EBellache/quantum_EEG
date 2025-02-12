import jax
from src.config import *

if USE_GPU:
    jax.config.update("jax_platform_name", "gpu")
    jax.config.update("jax_xla_backend", "cuda")

# Enable mixed precision
if ENABLE_MIXED_PRECISION:
    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_default_dtype_bits", 16)
