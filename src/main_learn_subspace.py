import sys, os
from functools import partial
import subprocess

import argparse, json

import numpy as np
import scipy
import scipy.optimize

import jax
import jax.numpy as jnp
import jax.scipy
import jax.scipy.optimize
import jax.nn
from jax.example_libraries import optimizers

import equinox as eqx

import polyscope as ps
import polyscope.imgui as psim

# import igl

# Imports from this project
import utils
import config
import layers
import subspace

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, "..")


def main():

    # Build command line arguments
    parser = argparse.ArgumentParser()

    # SEE config.py FOR WHERE MOST ARGS ARE ADDED

    # Shared arguments
    config.add_system_args(parser)
    config.add_learning_args(parser)
    config.add_training_args(parser)
    config.add_jax_args(parser)

    ###  Arguments specific to this program
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--output_prefix", type=str, default="")

    # network defaults
    parser.add_argument("--model_type", type=str, default='SubspaceMLP')
    parser.add_argument("--activation", type=str, default='ELU')

    # subspace
    parser.add_argument("--subspace_dim", type=int, default=6)
    parser.add_argument("--subspace_domain_type", type=str, default='normal')

    # loss weights / style / params
    parser.add_argument("--sigma_scale", type=float)
    parser.add_argument("--weight_expand", type=float)
    parser.add_argument("--expand_type", type=str, default="iso")

    # Parse arguments
    args = parser.parse_args()

    # Process args
    config.process_jax_args(args)

    # Force jax to initialize itself so errors get thrown early
    _ = jnp.zeros(())

    # some random state
    rngkey = jax.random.PRNGKey(0)

    # Build the system object
    system, system_def = config.construct_system_from_name(args.system_name, args.problem_name)

    # Subspace domain
    subspace_domain_dict = subspace.get_subspace_domain_dict(args.subspace_domain_type)

    # Build an informative output name

    network_filename_base = f"{args.output_prefix}neural_subspace_{args.system_name}_{args.problem_name}_{args.subspace_domain_type}_dim{args.subspace_dim}_wexp{args.weight_expand:g}"

    utils.ensure_dir_exists(args.output_dir)

    # configure other parameters
    target_dim = system.dim
    base_state = system_def['interesting_states'][0, :]
    rngkey, subkey = jax.random.split(rngkey)

    def sample_subspace(n_sample, rngkey):
        return subspace_domain_dict['domain_sample_fn']((n_sample, args.subspace_dim), rngkey)


    # Construct the learned subspace operator
    in_dim = args.subspace_dim + system.cond_dim
    model_spec = layers.model_spec_from_args(args, in_dim, target_dim)
    if subspace_domain_dict['domain_name'] == 'torus':
        model_spec['encode_periodic'] = True
    rngkey, subkey = jax.random.split(rngkey)
    model = layers.create_model(model_spec, subkey, base_output=base_state)
    model_params, model_static = eqx.partition(model, eqx.is_array)

    # Dictionary of other parameters which *are* updated by the optimizer
    # other_params = {'base_state': base_state}
    other_params = {}

    # Dictionary of extra external parameters which are non-constant but not updated by the optimizer
    ex_params = {'t_schedule': jnp.array(0.0)}

    @jax.jit
    def apply_subspace(model_params, x, cond_params, t_schedule):
        model = eqx.combine(model_params, model_static)
        return model(jnp.concatenate((x, cond_params), axis=-1), t_schedule=t_schedule)

    def mollify_norm(x, eps=1e-20):
        return jnp.sqrt(jnp.sum(jnp.square(x)) + eps)

    # Evaluate
    def sample_system_and_Epot(system_def, model_params, other_params, ex_params, rngkey):
        system_def = system_def.copy()
        t_schedule = ex_params['t_schedule']

        # Sample a conditional values
        rngkey, subkey = jax.random.split(rngkey)
        cond_params = system.sample_conditional_params(system_def, subkey, rho=t_schedule)
        system_def['cond_param'] = cond_params

        subspace_f = lambda zz: apply_subspace(model_params, zz, cond_params, t_schedule)

        # Sample latent value
        rngkey, subkey = jax.random.split(rngkey)
        z = sample_subspace(1, subkey)[0, :]

        # Map the latent state to config space
        q = subspace_f(z)
        E_pot = system.potential_energy(system_def, q)

        return z, cond_params, q, E_pot

    def batch_repulsion(z_batch, q_batch, t_schedule):
        # z_batch: [B,Z]
        # q_batch: [B,Q]
        # Returns [B] vector sum along columns
        DIST_EPS = 1e-8

        stats = {}

        def q_dists_one(q):
            return jax.vmap(partial(system.kinetic_energy, system_def,
                                    q))(q_batch - q[None, :]) + DIST_EPS

        all_q_dists = jax.vmap(q_dists_one)(q_batch)  # [B,B]

        def z_dists_one(z):
            z_delta = z_batch - z[None, :]
            return jnp.sum(z_delta*z_delta,axis=-1)

        all_z_dists = jax.vmap(z_dists_one)(z_batch)  # [B,B]

        if args.expand_type == 'iso':
            
            factor = jnp.log(t_schedule * args.sigma_scale * all_z_dists + DIST_EPS) - jnp.log(all_q_dists)
            repel_term = jnp.sum(jnp.square(0.5*factor), axis=-1)

            stats['mean_scale_log'] = jnp.mean(-factor)

        else:
            raise ValueError("expand type should be 'iso'")


        return repel_term, stats

    # Create an optimizer
    print(f"Creating optimizer...")

    def step_func(i_iter):
        out = args.lr * (args.lr_decay_frac**(i_iter // args.lr_decay_every))
        return out

    opt = optimizers.adam(step_func)
    all_params = (model_params, other_params)
    opt_state = opt.init_fn(all_params)

    def batch_loss_fn(params, ex_params, rngkey):

        model_params, other_params = params
        t_schedule = ex_params['t_schedule']

        subkey_b = jax.random.split(rngkey, args.batch_size)
        z_samples, cond_samples, q_samples, E_pots = jax.vmap(
            partial(sample_system_and_Epot, system_def, model_params, other_params, 
                    ex_params))(subkey_b)

        expand_loss, repel_stats = batch_repulsion(z_samples, q_samples, t_schedule)
        expand_loss = expand_loss * args.weight_expand

        loss_dict = {}
        loss_dict['E_pot'] = E_pots
        loss_dict['E_expand'] = expand_loss

        out_stats_b = {}
        out_stats_b.update(repel_stats)

        # sum up a total loss (mean over batch)
        total_loss = 0.
        for _, v in loss_dict.items():
            total_loss += jnp.mean(v)

        return total_loss, (loss_dict, out_stats_b)

    @jax.jit
    def train_step(i_iter, rngkey, ex_params, opt_state):

        opt_params = opt.params_fn(opt_state)
        (value, (loss_dict, out_stats_b)), grads = jax.value_and_grad(batch_loss_fn,
                                                                      has_aux=True)(opt_params,
                                                                                    ex_params,
                                                                                    rngkey)
        opt_state = opt.update_fn(i_iter, grads, opt_state)

        # out_stats_b currently unused

        return value, loss_dict, opt_state, out_stats_b

    print(f"Training...")
    last_print_iter = 0

    # Parameters tracked for each stat round
    losses = []
    n_sum_total = 0
    loss_term_sums = {}
    i_save = 0
    mean_scale_log = []

    ## Main training loop
    for i_train_iter in range(args.n_train_iters):

        ex_params['t_schedule'] = i_train_iter / args.n_train_iters

        rngkey, subkey = jax.random.split(rngkey)
        loss, loss_dict, opt_state, out_stats = train_step(i_train_iter, subkey, ex_params,
                                                           opt_state)

        # track statistics
        loss = float(loss)
        losses.append(loss)
        if 'mean_scale_log' in out_stats:
            mean_scale_log.append(out_stats['mean_scale_log'])

        for k in loss_dict:
            if k not in loss_term_sums:
                loss_term_sums[k] = 0.
            loss_term_sums[k] += jnp.sum(loss_dict[k])

        n_sum_total += args.batch_size

        def save_model(this_name):

            network_filename_pre = os.path.join(args.output_dir, network_filename_base) + this_name

            print(f"Saving result to {network_filename_pre}")

            model = eqx.combine(model_params, model_static)
            eqx.tree_serialise_leaves(network_filename_pre + ".eqx", model)
            with open(network_filename_pre + '.json', 'w') as json_file:
                json_file.write(json.dumps(model_spec))
            np.save(
                network_filename_pre + "_info", {
                    'system': args.system_name,
                    'problem_name': args.problem_name,
                    'subspace_domain_type': args.subspace_domain_type,
                    'subspace_dim': args.subspace_dim,
                    't_schedule_final': ex_params['t_schedule'],
                })

            print(f"  ...done saving")
                

        if i_train_iter % args.report_every == 0:

            print(
                f"\n== iter {i_train_iter} / {args.n_train_iters}  ({100.*i_train_iter/args.n_train_iters:.2f}%)"
            )


            # print some statistics
            mean_loss = np.mean(np.array(losses))
            print(f"      loss: {mean_loss:.6f}")

            opt_params = opt.params_fn(opt_state)
            model_params, other_params = opt_params

            for k in loss_term_sums:
                print(f"   {k:>30}: {(loss_term_sums[k] / n_sum_total):.6f}")

            print("  Stats:")
            if len(mean_scale_log) > 0:
                mean_scale = jnp.exp(jnp.mean(jnp.stack(mean_scale_log)))
                print(
                    f"    mean metric stretch: {mean_scale:g}    (scaled so current target is 1.)")
            

            # save
            out_name = f"_save{i_save:04d}"
            if args.expand_type == 'iso':
                scale_combined = args.sigma_scale * ex_params['t_schedule']
                out_name += f'_sigma{scale_combined:g}'
            save_model(out_name)
            i_save += 1

            # reset statistics
            losses = []
            n_sum_total = 0
            loss_term_sums = {}
            mean_scale_log = []

    # save results one last time
    save_model("_final")


if __name__ == '__main__':
    main()
