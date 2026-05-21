import time
import numpy as np
import jax.numpy as jnp
import jax
from hj_reachability import sets
from hj_reachability.grid import Grid

# Import our fast interpolation methods (this will monkey-patch them onto Grid)
import valtr.faster_hj_grid_interpolation

def create_test_grid(ndim, shape_per_dim=20):
    """Create a test grid with specified dimensions."""
    domain = sets.Box(
        lo=jnp.array([-1.0] * ndim),
        hi=jnp.array([1.0] * ndim)
    )
    shape = tuple([shape_per_dim] * ndim)
    return Grid.from_lattice_parameters_and_boundary_conditions(domain, shape)

def run_quick_test():
    """Quick test to verify the methods work."""
    print("=== Quick Test ===")
    
    # Create a simple 3D grid
    grid = create_test_grid(3, shape_per_dim=10)
    values = jnp.sum(grid.states**2, axis=-1)  # Simple test function
    test_state = jnp.array([0.5, 0.3, -0.2])
    
    print(f"Grid shape: {grid.shape}")
    print(f"Test state: {test_state}")
    
    # Test all methods
    try:
        result_original = grid.interpolate(values, test_state)
        print(f"Original interpolate: {result_original}")
        
        result_fast = grid.interpolate_fast(values, test_state)
        print(f"Fast interpolate: {result_fast}")
        
        result_fast_jit = grid.interpolate_fast_jit(values, test_state)
        print(f"Fast JIT interpolate: {result_fast_jit}")
        
        # Test batch method
        test_states = jnp.array([test_state, test_state * 0.8, test_state * 1.2])
        result_batch = grid.interpolate_fast_batch_jit(values, test_states)
        print(f"Batch JIT interpolate: {result_batch}")
        
        # Check accuracy
        diff_fast = abs(result_original - result_fast)
        diff_jit = abs(result_original - result_fast_jit)
        
        print(f"\nAccuracy check:")
        print(f"  Original vs Fast: {diff_fast} (should be < 1e-10)")
        print(f"  Original vs JIT: {diff_jit} (should be < 1e-6)")  # Relaxed tolerance for JIT
        
        if diff_fast < 1e-10 and diff_jit < 1e-6:  # Relaxed tolerance for JIT
            print("✓ All methods produce identical results (within numerical precision)!")
            return True
        else:
            print("✗ Methods produce different results!")
            return False
                
    except Exception as e:
        print(f"Error during testing: {e}")
        return False

if __name__ == "__main__":
    success = run_quick_test()
    if success:
        print("\nAll tests passed! You can now use the fast interpolation methods.")
    else:
        print("\nSome tests failed. Check the implementation.")
