import jax
import jax.numpy as jnp


def generate_fixed_entry_data(fixed_mask, initial_values):
    '''
    Preprocess to construct data to efficiently apply fixed values (e.g. pinned vertices, boundary conditions) to a vector.

    Suppose we want a vector of length N, with F fixed entries.
        - `fixed_mask` is a length-N boolean array, where `True` entries are fixed
        - `initial_values` is a length-N float array giving the fixed values. Elements corresponding to non-fixed entries are ignored.
    '''
    
    fixed_inds = jnp.nonzero(fixed_mask)[0]
    unfixed_inds = jnp.nonzero(~fixed_mask)[0]
    fixed_values = initial_values[fixed_mask]
    unfixed_values = initial_values[~fixed_mask]

    return fixed_inds, unfixed_inds, fixed_values, unfixed_values

def apply_fixed_entries(fixed_inds, unfixed_inds, fixed_values, unfixed_values):
    '''
    Applies fixed values to a vector, using the indexing arrays generated from generate_fixed_entry_data(), plus a vector of all the un-fixed values.

    Passing fixed_values or unfixed_values as a scalar will also work.
    '''

    out = jnp.zeros(fixed_inds.shape[0] + unfixed_inds.shape[0])
    out = out.at[fixed_inds].set(fixed_values, indices_are_sorted=True, unique_indices=True)
    out = out.at[unfixed_inds].set(unfixed_values, indices_are_sorted=True, unique_indices=True)

    return out
