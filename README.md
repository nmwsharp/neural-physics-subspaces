## Data-Free Learning of Reduced-Order Kinematics (SIGGRAPH 2023)
 
**Authors:** Nicholas Sharp, Cristian Romero, Alec Jacobson, Etienne Vouga, Paul Kry, David I.W. Levin, Justin Solomon

[[project page]](https://nmwsharp.com/research/neural-physics-subspaces/)  [[PDF]](https://nmwsharp.com/media/papers/neural-physics-subspaces/neural_physics_subspaces.pdf)  [[video]](https://youtu.be/6X2inurzkrs)

Use neural networks to fit low-dimensional subspaces for simulations, with no dataset needed—the method automatically explores the potential energy landscape.

This repo contains an implementation of our core procedure on some sample physical systems.

## Installing

This repository is standard Python code, tested with Python 3.9 on Ubuntu 20.04, as well as OSX systems.

The most significant depenency is JAX, which can be installed according to instructions here: https://jax.readthedocs.io/en/latest/

Other dependencies are all available through pip and conda. Conda `environment.yml` file is included to help resolve dependencies.

This code runs on CPUs or GPUs, although generally the GPU will be dramatically faster.

## Running

### Basic

To run our systems and visualize them in a GUI, call

```
python src/main_run_system.py --system_name [system_name] --problem_name [problem_name]
```
where the available systems and problems are given below. This will run the system in the full configuration space. Click on UI elements in the upper-right to run the dynamics, etc. Note that the triangular tree nodes can be expanded to yield more options.

### Fitting

To fit a subspace, call the following function. See the paper for parameters for the examples we showed.

```
python src/main_learn_subspace.py --system_name [system_name] --problem_name [problem_name] --subspace_dim=8 --weight_expand=1.0 --sigma_scale=1.0 --output_dir output/
```

which will iterative train and dump the resulting network files to `output/`

### Running fitted models

Once the subspace has been fitted, it can be loaded in to the run script as

```
python src/main_run_system.py --system_name [system_name] --problem_name [problem_name] --subspace [subspace_file_prefix]
```
to explore the subspace. Note that [subspace_file_prefix] should not be a full filename, but the prefix which is printed by the training script.

Three quick notes about apparent performance:
   - JAX code is JIT-compiled, so all routines will have a lag when they are called for the first time.
   - JAX by default allocates nearly all available GPU memory at startup.
   - This vesion of our GUI is implemented in a way which may be slow on some machines. Be wary that rendering the scene data may actually be dominating the runtime, for incidental software reasons unrelated to our algorithm. Alternate GUIs can be used to circumvent the issue if needed.


## Physical systems

The following systems are included:

- "FEM" -- finite elemnet deformables in 2d and 3d
  - "bistable" -- 2d bistable bar
  - "load3d" -- 3d cantilevered bar
  - "heterobeam" -- 3d cantilevered bar with non-constant stiffness
- "rigid3d" -- rigid bodies in 3d
  - "klann" -- Klann linkage
  - "stewart" -- Stewart mechansim
