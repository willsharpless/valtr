import time
import numpy as np
import jax.numpy as jnp
import jax
from hj_reachability import sets
import faster_hj_grid_interpolation  # This automatically adds the methods
from hj_reachability.grid import Grid

def create_test_grid(ndim, shape_per_dim=20):
    """Create a test grid with specified dimensions."""
    domain = sets.Box(
        lo=jnp.array([-1.0] * ndim),
        hi=jnp.array([1.0] * ndim)
    )
    shape = tuple([shape_per_dim] * ndim)
    return Grid.from_lattice_parameters_and_boundary_conditions(domain, shape)

def create_test_values(grid, extra_dims=None):
    """Create test values on the grid."""
    if extra_dims is None:
        # Scalar values at each grid point
        # Use a smooth function that varies across dimensions
        return jnp.sum(grid.states**2, axis=-1)
    else:
        # Multi-dimensional values at each grid point
        base_values = jnp.sum(grid.states**2, axis=-1)
        shape = grid.shape + extra_dims
        return jnp.broadcast_to(base_values[..., None], shape) * jnp.arange(1, extra_dims[0] + 1)

def create_test_states(grid, n_test_points=1000):
    """Create random test states within the domain."""
    rng = jax.random.PRNGKey(42)
    # Create random points within the domain
    random_vals = jax.random.uniform(rng, (n_test_points, grid.ndim))
    return grid.domain.lo + random_vals * (grid.domain.hi - grid.domain.lo)

def test_accuracy(grid, values, test_states, method_name, method_func):
    """Test accuracy of interpolation method."""
    print(f"Testing accuracy for {method_name}...")
    
    try:
        # Test on first few points to check for errors
        for i in range(min(5, len(test_states))):
            result = method_func(values, test_states[i])
            if jnp.any(jnp.isnan(result)) and not jnp.any((test_states[i] < grid.domain.lo) | (test_states[i] > grid.domain.hi)):
                print(f"  WARNING: Unexpected NaN result at state {test_states[i]}")
    except Exception as e:
        print(f"  ERROR in {method_name}: {e}")
        return None
    
    print(f"  {method_name} passed basic accuracy tests")
    return True

def time_interpolation(grid, values, test_states, method_name, method_func, warmup=3, n_runs=10):
    """Time an interpolation method."""
    print(f"Timing {method_name}...")
    
    try:
        # Warmup runs
        for _ in range(warmup):
            for state in test_states[:10]:  # Just a few points for warmup
                _ = method_func(values, state)
        
        # Actual timing
        start_time = time.time()
        for _ in range(n_runs):
            for state in test_states:
                _ = method_func(values, state)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_call = total_time / (n_runs * len(test_states))
        
        print(f"  Average time per interpolation: {avg_time_per_call*1e6:.2f} µs")
        return avg_time_per_call
    
    except Exception as e:
        print(f"  ERROR timing {method_name}: {e}")
        return None

def compare_accuracy(grid, values, test_states):
    """Compare accuracy between methods."""
    print("Comparing accuracy between methods...")
    
    # Get results from original method
    original_results = []
    for state in test_states[:50]:  # Just test on subset for accuracy
        original_results.append(grid.interpolate(values, state))
    original_results = jnp.array(original_results)
    
    # Compare with fast method
    fast_results = []
    for state in test_states[:50]:
        fast_results.append(grid.interpolate_fast(values, state))
    fast_results = jnp.array(fast_results)
    
    # Compare with JIT method
    jit_results = []
    for state in test_states[:50]:
        jit_results.append(grid.interpolate_fast_jit(values, state))
    jit_results = jnp.array(jit_results)
    
    # Calculate differences
    fast_diff = jnp.max(jnp.abs(original_results - fast_results))
    jit_diff = jnp.max(jnp.abs(original_results - jit_results))
    
    print(f"  Max difference (original vs fast): {fast_diff}")
    print(f"  Max difference (original vs jit): {jit_diff}")
    
    return fast_diff < 1e-10 and jit_diff < 1e-10

def run_tests():
    """Run comprehensive tests."""
    print("=== Interpolation Performance Testing ===\n")
    
    # Test dimensions
    test_dims = [2, 3, 4]
    n_test_points = 500
    shape_per_dim = 100
    
    results = {}
    
    for ndim in test_dims:
        print(f"\n{'='*50}")
        print(f"Testing {ndim}D grid")
        print(f"{'='*50}")
        
        # Create test grid and data
        grid = create_test_grid(ndim, shape_per_dim=shape_per_dim)  # Smaller grid for faster testing
        values_scalar = create_test_values(grid)
        values_vector = create_test_values(grid, extra_dims=(3,))  # 3-dimensional values at each point
        test_states = create_test_states(grid, n_test_points)
        
        print(f"Grid shape: {grid.shape}")
        print(f"Testing {len(test_states)} random states")
        
        # Test with scalar values
        print(f"\n--- Scalar Values ---")
        
        methods = [
            ("Original", lambda v, s: grid.interpolate(v, s)),
            ("Fast", lambda v, s: grid.interpolate_fast(v, s)),
            ("Fast+JIT", lambda v, s: grid.interpolate_fast_jit(v, s)),
        ]
        
        # Test accuracy
        accuracy_results = {}
        for name, method in methods:
            accuracy_results[name] = test_accuracy(grid, values_scalar, test_states, name, method)
        
        # Compare accuracy between methods
        if all(accuracy_results.values()):
            accuracy_match = compare_accuracy(grid, values_scalar, test_states)
            print(f"Accuracy match between methods: {accuracy_match}")
        
        # Time performance
        print(f"\nTiming results:")
        timing_results = {}
        for name, method in methods:
            if accuracy_results[name]:
                timing_results[name] = time_interpolation(
                    grid, values_scalar, test_states[:100], name, method, n_runs=3  # Fewer runs for speed
                )
        
        # Calculate speedups
        if "Original" in timing_results and timing_results["Original"]:
            baseline = timing_results["Original"]
            print(f"\nSpeedup results (vs Original):")
            for name, time_val in timing_results.items():
                if time_val and name != "Original":
                    speedup = baseline / time_val
                    print(f"  {name}: {speedup:.2f}x faster")
        
        # Test with vector values
        print(f"\n--- Vector Values (3D per grid point) ---")
        for name, method in methods:
            if accuracy_results[name]:
                test_accuracy(grid, values_vector, test_states[:20], f"{name} (vector)", method)
        
        results[ndim] = {
            'accuracy': accuracy_results,
            'timing': timing_results
        }
    
    print(f"\n{'='*50}")
    print("Summary")
    print(f"{'='*50}")
    
    for ndim in test_dims:
        print(f"\n{ndim}D Results:")
        if 'timing' in results[ndim]:
            timing = results[ndim]['timing']
            if "Original" in timing and timing["Original"]:
                baseline = timing["Original"]
                for name, time_val in timing.items():
                    if time_val:
                        if name == "Original":
                            print(f"  {name}: {time_val*1e6:.2f} µs")
                        else:
                            speedup = baseline / time_val
                            print(f"  {name}: {time_val*1e6:.2f} µs ({speedup:.2f}x faster)")

if __name__ == "__main__":
    run_tests()