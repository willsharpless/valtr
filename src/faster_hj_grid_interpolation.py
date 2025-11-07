import jax
from hj_reachability.grid import Grid
import jax.numpy as jnp
import functools

# Create JIT-compiled functions for specific dimensions
def _make_jit_interpolate_for_ndim(ndim):
    """Create a JIT-compiled interpolate function for a specific number of dimensions."""
    corner_selectors = jnp.array([[bool((i >> j) & 1) for j in range(ndim)] 
                                 for i in range(2**ndim)])
    
    @jax.jit
    def _jit_helper(values, state, domain_lo, spacings, shape, is_periodic_dim):
        spacings_array = jnp.array(spacings)
        shape_array = jnp.array(shape)
        
        position = (state - domain_lo) / spacings_array
        index_lo = jnp.floor(position).astype(jnp.int32)
        index_hi = index_lo + 1
        weight_hi = position - index_lo
        weight_lo = 1 - weight_hi
        
        # Handle boundaries
        index_lo = jnp.where(is_periodic_dim, index_lo % shape_array, 
                            jnp.clip(index_lo, 0, shape_array - 1))
        index_hi = jnp.where(is_periodic_dim, index_hi % shape_array, 
                            jnp.clip(index_hi, 0, shape_array - 1))
        
        # Use pre-computed corner selectors
        all_indices = jnp.where(corner_selectors, 
                                index_hi[None, :], 
                                index_lo[None, :]).astype(jnp.int32)
        
        all_weights = jnp.prod(jnp.where(corner_selectors, 
                                        weight_hi[None, :], 
                                        weight_lo[None, :]), axis=1)
        
        # Use more JAX-friendly indexing approach
        # Manual loop unrolling for better consistency with non-JIT version
        def get_corner_value(i):
            idx = tuple(all_indices[i])
            return values[idx]
        
        corner_values = jnp.stack([get_corner_value(i) for i in range(2**ndim)])
        
        # Handle multidimensional values correctly
        if values.ndim > ndim:
            # Reshape weights to broadcast properly with multidimensional values
            extra_dims = values.ndim - ndim
            weight_shape = (2**ndim,) + (1,) * extra_dims
            all_weights = all_weights.reshape(weight_shape)
            result = jnp.sum(all_weights * corner_values, axis=0)
        else:
            result = jnp.sum(all_weights * corner_values)
        
        # Boundary check
        domain_hi = domain_lo + spacings_array * (shape_array - 1)
        out_of_bounds = jnp.any(~is_periodic_dim & 
                                ((state < domain_lo) | (state > domain_hi)))
        return jnp.where(out_of_bounds, jnp.nan, result)
    
    return _jit_helper

# Pre-compile for common dimensions
_jit_helpers = {
    2: _make_jit_interpolate_for_ndim(2),
    3: _make_jit_interpolate_for_ndim(3),
    4: _make_jit_interpolate_for_ndim(4),
    5: _make_jit_interpolate_for_ndim(5),
    6: _make_jit_interpolate_for_ndim(6),
}

def interpolate_fast(self, values, state):
    """Interpolates `values` (possibly multidimensional per node) defined over the grid at the given `state`."""
    spacings_array = jnp.array(self.spacings)
    shape_array = jnp.array(self.shape)
    
    position = (state - self.domain.lo) / spacings_array
    index_lo = jnp.floor(position).astype(jnp.int32)
    index_hi = index_lo + 1
    weight_hi = position - index_lo
    weight_lo = 1 - weight_hi
    
    # Handle boundaries
    if jnp.any(self._is_periodic_dim):
        index_lo = jnp.where(self._is_periodic_dim, index_lo % shape_array, 
                        jnp.clip(index_lo, 0, shape_array - 1))
        index_hi = jnp.where(self._is_periodic_dim, index_hi % shape_array, 
                        jnp.clip(index_hi, 0, shape_array - 1))
    else:
        index_lo = jnp.clip(index_lo, 0, shape_array - 1)
        index_hi = jnp.clip(index_hi, 0, shape_array - 1)
    
    # Vectorized approach: create all corner combinations at once
    corner_selectors = jnp.array([[bool((i >> j) & 1) for j in range(self.ndim)] 
                                for i in range(2**self.ndim)])
    
    # Vectorized index selection: (2^ndim, ndim)
    all_indices = jnp.where(corner_selectors, 
                        index_hi[None, :], 
                        index_lo[None, :]).astype(jnp.int32)
    
    # Vectorized weight calculation: (2^ndim,)
    all_weights = jnp.prod(jnp.where(corner_selectors, 
                                    weight_hi[None, :], 
                                    weight_lo[None, :]), axis=1)
    
    # Use consistent indexing approach with JIT version
    def get_corner_value(i):
        idx = tuple(all_indices[i])
        return values[idx]
    
    corner_values = jnp.stack([get_corner_value(i) for i in range(2**self.ndim)])
    
    # Handle multidimensional values correctly
    if values.ndim > self.ndim:
        # Reshape weights to broadcast properly with multidimensional values
        extra_dims = values.ndim - self.ndim
        weight_shape = (2**self.ndim,) + (1,) * extra_dims
        all_weights = all_weights.reshape(weight_shape)
        result = jnp.sum(all_weights * corner_values, axis=0)
    else:
        result = jnp.sum(all_weights * corner_values)
    
    # Boundary check
    out_of_bounds = jnp.any(~self._is_periodic_dim & 
                        ((state < self.domain.lo) | (state > self.domain.hi)))
    return jnp.where(out_of_bounds, jnp.nan, result)

def interpolate_fast_jit(self, values, state):
    """JIT-compiled version using dimension-specific helper function."""
    if self.ndim in _jit_helpers:
        return _jit_helpers[self.ndim](
            values, state, self.domain.lo, self.spacings, self.shape, 
            self._is_periodic_dim
        )
    else:
        # Fall back to non-JIT version for unsupported dimensions
        print(f"Warning: JIT not supported for {self.ndim}D grids, using non-JIT version")
        return self.interpolate_fast(values, state)

def interpolate_fast_batch_jit(self, values, states):
    """Batch interpolation for multiple states using JIT."""
    if self.ndim in _jit_helpers:
        def single_interpolate(state):
            return _jit_helpers[self.ndim](
                values, state, self.domain.lo, self.spacings, self.shape, 
                self._is_periodic_dim
            )
        
        # Use vmap for efficient batch processing
        return jax.vmap(single_interpolate)(states)
    else:
        # Fall back to non-JIT version
        print(f"Warning: JIT not supported for {self.ndim}D grids, using non-JIT version")
        return jax.vmap(lambda state: self.interpolate_fast(values, state))(states)

# Monkey-patch the methods onto the Grid class
def add_fast_interpolation_methods():
    """Add fast interpolation methods to the Grid class."""
    Grid.interpolate_fast = interpolate_fast
    Grid.interpolate_fast_jit = interpolate_fast_jit
    Grid.interpolate_fast_batch_jit = interpolate_fast_batch_jit

# Automatically add the methods when this module is imported
add_fast_interpolation_methods()