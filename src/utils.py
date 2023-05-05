import time
import os
import inspect

from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

import polyscope as ps
import polyscope.imgui as psim

def ensure_dir_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)

def dict_to_tuple(d):
    if d is None: return None
    return tuple(sorted(d.items()))

def tuple_to_dict(t):
    if t is None: return None
    d = {}
    for k,v in t:
        d[k] = v
    return d

# === Polyscope

def combo_string_picker(name, curr_val, vals_list):

    changed = psim.BeginCombo(name, curr_val)
    clicked = False
    if changed:
        for val in vals_list:
            _, selected = psim.Selectable(val, curr_val==val)
            if selected:
                curr_val=val
                clicked = True
        psim.EndCombo()

    return clicked, curr_val

# === JAX helpers

# quick printing
def printarr(*arrs, data=True, short=True, max_width=200):
    
    frame = inspect.currentframe().f_back
    default_name = "[unnamed]"

    # helpers for below
    def compress_str(s):
        return s.replace('\n', '')
    def dtype_str(a):
        if a is None:
            return 'N/A'
        if isinstance(a, int):
            return 'python-int'
        if isinstance(a, float):
            return 'python-float'
        return str(a.dtype)
    def shape_str(a):
        if a is None:
            return 'N/A'
        if isinstance(a, int):
            return 'scalar'
        if isinstance(a, float):
            return 'scalar'
        return str(a.shape)
    def name_from_outer_scope(a):
        if a is None:
            return '[None]'

        name = default_name
        for k, v in frame.f_locals.items():
            if v is a:
                name = k
                break
        return name


    name_align = ">" if short else "<"

    # get the name of the tensor as a string
    try:
        # first compute some length stats
        name_len = -1
        dtype_len = -1
        shape_len = -1
        for a in arrs:
            name = name_from_outer_scope(a)
            name_len = max(name_len, len(name))
            dtype_len = max(dtype_len, len(dtype_str(a)))
            shape_len = max(shape_len, len(shape_str(a)))
        len_left = max_width - name_len - dtype_len - shape_len - 5

        # now print the acual arrays
        for a in arrs:
            name = name_from_outer_scope(a)
            print(
                f"{name:{name_align}{name_len}} {dtype_str(a):<{dtype_len}} {shape_str(a):>{shape_len}}",
                end='')
            if data:
                # print the contents of the array
                print(": ", end='')
                flat_str = compress_str(str(a))
                if len(flat_str) < len_left:
                    # short arrays are easy to print
                    print(flat_str)
                else:
                    # long arrays
                    if short:
                        # print a shortented version that fits on one line
                        if len(flat_str) > len_left - 4:
                            flat_str = flat_str[:(len_left - 4)] + " ..."
                        print(flat_str)
                    else:
                        # print the full array on a new line
                        print("")
                        print(a)
            else:
                print("")  # newline
    finally:
        del frame


# def convert_to_dtype(pytree, dtype):
#     def convert_func(x):
#         if isinstance(jnp.array) and x.dtype != dtype:
#             x.to(dtype)
#         else:
#             return x

def logical_and_all(vals):
    out = vals[0]
    for i in range(1,len(vals)):
        out = jnp.logical_and(out, vals[i])
    return out

def logical_or_all(vals):
    out = vals[0]
    for i in range(1,len(vals)):
        out = jnp.logical_or(out, vals[i])
    return out

def minimum_all(vals):
    '''
    Take elementwise minimum of a list of arrays
    '''
    combined = jnp.stack(vals, axis=0)
    return jnp.min(combined, axis=0)

def maximum_all(vals):
    '''
    Take elementwise maximum of a list of arrays
    '''
    combined = jnp.stack(vals, axis=0)
    return jnp.max(combined, axis=0)

def all_same_sign(vals):
    '''
    Test if all values in an array have (strictly) the same sign
    '''
    return jnp.logical_or(jnp.all(vals < 0), jnp.all(vals > 0))

# Given a 1d array mask, enumerate the nonero entries 
# example:
# in:  [0 1 1 0 1 0]
# out: [X 0 1 X 2 X]
# where X = fill_val
# if fill_val is None, the array lenght + 1 is used
def enumerate_mask(mask, fill_value=None):
    if fill_value is None:
        fill_value = mask.shape[-1]+1
    out = jnp.cumsum(mask, axis=-1)-1
    out = jnp.where(mask, out, fill_value)
    return out


# Returns the first index past the last True value in a mask
def empty_start_ind(mask):
    return jnp.max(jnp.arange(mask.shape[-1]) * mask)+1
    
# Given a list of arrays all of the same shape, interleaves
# them along the first dimension and returns an array such that
# out.shape[0] = len(arrs) * arrs[0].shape[0]
def interleave_arrays(arrs):
    s = list(arrs[0].shape)
    s[0] *= len(arrs)
    return jnp.stack(arrs, axis=1).reshape(s)

@partial(jax.jit, static_argnames=("new_size","axis"))
def resize_array_axis(A, new_size, axis=0):
    first_N = min(new_size, A.shape[0])
    shape = list(A.shape)
    shape[axis] = new_size
    new_A = jnp.zeros(shape, dtype=A.dtype)
    new_A = new_A.at[:first_N,...].set(A.at[:first_N,...].get())
    return new_A

def smoothstep(x):
    out = 3.*x*x - 2.*x*x*x
    out = jnp.where(x < 0, 0., out)
    out = jnp.where(x > 1, 1., out)
    return out

def binary_cross_entropy_loss(logit_in, target):
    # same as the pytorch impl, allegedly numerically stable
    neg_abs = -jnp.abs(logit_in)
    loss = jnp.clip(logit_in, a_min=0) - logit_in * target + jnp.log(1 + jnp.exp(neg_abs))
    return loss

def normalize(X, eps=0.):
    # normalizes along last dimension
    denom = jnp.linalg.norm(X, axis=-1, keepdims=True)
    return X / (denom + eps)

def torus_dist2(x,y):
    # "euclidean" distance^2 in the flat [0,1] torus
    d = jnp.abs(x-y)
    d = jnp.where(d > 0.5, 1.0 - d, d)
    return jnp.sum(d*d,axis=-1); 

