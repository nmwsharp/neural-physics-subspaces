from functools import partial

import numpy as np
import scipy.optimize
import jax
import jax.numpy as jnp
import jax.scipy.optimize


methods = [
    'JAX-LBFGS',
]

# solve
# returns a tuple of: (result, solver_info)
# where solver_info is a dictionary of solver-defined statistics
def minimize_nonlinear(func, init_guess, method):

    if method == 'JAX-LBFGS':
        res = jax.scipy.optimize.minimize(func,
                                          init_guess,
                                          method='l-bfgs-experimental-do-not-rely-on-this',
                                          options={
                                              'maxiter': 1024,
                                              'maxls': 128
                                          })
        return (res.x, {
            "success": res.success,
            "scipy_optimize_status": res.status,
            "n_iter": res.nit,
        })

    else:
        raise ValueError("unrecognized method")


def print_solver_info(info):
    print("  solver info:")
    for k in info:
        val = info[k]
        if k in ['success']:
            val = bool(val)
        if k in ['n_iter']:
            val = int(val)
        if k in ['scipy_optimize_status']:
            val = {
                0: "converged (nominal)",
                1: "max BFGS iters reached",
                3: "zoom failed",
                4: "saddle point reached",
                5: "line search failed",
                -1: "undefined"
            }[int(val)]
        print(f"   {k:>30}: {val}")
