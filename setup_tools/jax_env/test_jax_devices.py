import jax
import jaxlib

print(jax.devices(), flush=True)
print(f"JAX: {jax.__version__} | jaxlib: {jaxlib.__version__}")
