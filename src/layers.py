from functools import partial
import typing

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import jaxtyping


def str_to_act(s):
    d = {
        'ReLU': jax.nn.relu,
        'LeakyReLU': jax.nn.leaky_relu,
        'ELU': jax.nn.elu,
        'Cos': jnp.cos,
    }

    if s not in d:
        raise ValueError(f'Unrecognized activation {s}. Should be one of {d.keys()}')

    return d[s]


def model_spec_from_args(args, in_dim, out_dim):
    """
    Build the dictionary which we feed into the more general create_network
    (this dictionary can be saved to recreate the same network later)
    """

    spec_dict = {}

    spec_dict['in_dim'] = in_dim
    spec_dict['out_dim'] = out_dim

    spec_dict['model_type'] = args.model_type

    # Add MLP-specific args
    if spec_dict['model_type'] in ["SubspaceMLP"]:

        spec_dict['activation'] = args.activation
        spec_dict['MLP_hidden_layers'] = args.MLP_hidden_layers
        spec_dict['MLP_hidden_layer_width'] = args.MLP_hidden_layer_width

    # Add linear-specific args
    elif spec_dict['model_type'] == "Linear":
        # TODO implement
        pass

    else:
        raise ValueError(f"unrecognized model_type {spec_dict['model_type']}")

    return spec_dict


def create_model(spec_dict, rngkey=None, base_output=None):

    if rngkey is None:
        rngkey = jax.random.PRNGKey(0)

    if spec_dict['model_type'] == "MLP":
        model = eqx.nn.MLP(spec_dict['in_dim'],
                           spec_dict['out_dim'],
                           spec_dict['MLP_hidden_layer_width'],
                           spec_dict['MLP_hidden_layers'],
                           activation=str_to_act(spec_dict['activation']),
                           key=rngkey)
    elif spec_dict['model_type'] == "MLP-Layers":
        model = MLP(spec_dict['in_dim'],
                    spec_dict['out_dim'],
                    [int(a) for a in spec_dict['hidden_layer_sizes'].split(',')],
                    activation=str_to_act(spec_dict['activation']),
                    rngkey=rngkey)
    elif spec_dict['model_type'] == "SubspaceMLP":
        model = SubspaceMLP(spec_dict, rngkey, base_output=base_output)
    elif spec_dict['model_type'] == "PeriodicMLP":
        model = PeriodicMLP(spec_dict, rngkey)
    elif spec_dict['model_type'] == "Linear":
        # TODO implement
        model = None
        pass

    else:
        raise ValueError(f"unrecognized model_type {spec_dict['model_type']}")

    # Always create the model in 32-bit, even if the system is defaulting to 64 bit.
    # Otherwise serialization things fail. Passing 64-bit inputs to the model should give
    # the expected up-conversion at evaluation time.
    model = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if eqx.is_array(x) else x, model)

    print(f"\n== Created network ({spec_dict['model_type']}):")
    print(model)

    return model



class SubspaceMLP(eqx.Module):

    linear_layers: typing.List[eqx.nn.Linear]
    activation: typing.Callable
    base_output: jaxtyping.Array

    def __init__(self, spec_dict, rngkey, base_output):

        # MLP layers
        self.activation = str_to_act(spec_dict['activation'])

        self.linear_layers = []
        prev_width = spec_dict['in_dim']
        for i_layer in range(spec_dict['MLP_hidden_layers']):
            is_last = (i_layer + 1 == spec_dict['MLP_hidden_layers'])
            next_width = spec_dict['out_dim'] if is_last else spec_dict['MLP_hidden_layer_width']

            rngkey, subkey = jax.random.split(rngkey)
            self.linear_layers.append(
                eqx.nn.Linear(prev_width, next_width, use_bias=True, key=subkey))
            prev_width = next_width

        rngkey, subkey = jax.random.split(rngkey)

        if base_output is None:
            self.base_output = jax.random.uniform(subkey, (spec_dict['out_dim'], ),
                                                  minval=-1.,
                                                  maxval=1.)
        else:
            self.base_output = base_output

    def __call__(self, z, t_schedule=1.):

        # MLP layes
        n_layers = len(self.linear_layers)
        for i_layer in range(n_layers):
            is_last = (i_layer + 1 == n_layers)

            z = self.linear_layers[i_layer](z)

            if not is_last:
                z = self.activation(z)

        z = self.base_output + t_schedule * z

        return z
