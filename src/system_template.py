import jax.numpy as jnp
from jax import grad, jit, vmap
import fixed_point_projection
import numpy as np
import polyscope as ps

import utils
import config

class TemplateSystem:

    @staticmethod
    def construct(problem_name):

        '''

        Basic philosophy:
            We define the system via two objects, a object instance ('system'), which can
            hold pretty much anything (strings, function pointers, etc), and a dictionary 
            ('system_def') which holds only jnp.array objects.

            The reason for this split is the JAX JIT system. Data stored in the `system`
            is fixed after construction, so it can be anything. Data stored in `system_def`
            can potentially be modified after the system is constructed, so it must consist
            only of JAX arrays to make JAX's JIT engine happy.


        The fields which MUST be populated are:
       
            = System name
            system.system_name --> str
            The name of the system (e.g. "neohookean")
            

            = Problem name
            system.problem_name --> str
            The name of the problem (e.g. "trussbar2")
       

            = Dimension
            system.dim --> int
                The dimension of the configuration space for the system. When we
                learn subspaces that map from `R^d --> R^n`, this is `n`. If the 
                system internally expands the configuration space e.g. to append 
                additional pinned vertex positions, those should NOT be counted 
                here.


            = Initial position
            system_def['init_pos'] --> jnp.array float dimension: (n,)
                An initial position, used to set the initial state in the GUI
                It does not necessarily need to be the rest pose of the system.
            

            = Conditional parameters dimension
            system.cond_dim --> int
                The dimension of the conditional parameter space for the system. 
                If there are no conditional parameters, use 0.
            
            = Conditional parameters value
            system_def['cond_param'] --> jnp.array float dimension: (c,)
                A vector of conditional paramters for the system, defining its current
                state. If there are no conditional parameters, use a length-0 vector.
            

            = External forces
            system_def['external_forces'] --> dictionary of anything
                Data defining external forces which can be adjusted at runtime
                The meaning of this dictionary is totally system-dependent. If unused, 
                leave as an empty dictionary. Fill the dictionary with arrays or other 
                data needed to evaluate external forces.


            = Interesting states
            system_def['interesting_states'] --> jnp.array float dimension (i,n)
                A collection of `i` configuration state vectors which we want to explicitly track
                and preserve.
                If there are no interesting states (i=0), then this should just be a size (0,n) array.


            The dictionary may also be populated with any other user-values which are useful in 
            defining the system.

            NOTE: When you add a new system class, also add it to the registry in config.py 
        '''

        system_def = {}


        # Example values:
        # (dummy data)
        system_def['system_name'] = "template_system"
        system_def['problem_name'] = problem_name
        system_def['cond_params'] = jnp.zeros((0,)) # a length-0 array
        system_def['external_forces'] = {}

        if problem_name == 'problem_A':
        
            config_dim = 280
            system_def['dim'] = config_dim
            system_def['init_pos'] = jnp.zeros(config_dim) # some values
            system_def['interesting_states'] = jnp.zeros((0,config_dim))
        
        elif problem_name == 'problem_B':

            # and so on....
            config_dim = 334
            system_def['dim'] = config_dim
            system_def['init_pos'] = jnp.zeros(config_dim) # some other values
            system_def['interesting_states'] = jnp.zeros((0,config_dim))

        else:
            raise ValueError("could not parse problem name: " + str(problem_name))


        return system_def
  
    # ===========================================
    # === Energy functions 
    # ===========================================

    # These define the core physics of our system

    def potential_energy(system_def, q):
        # TODO implement
        return 0.
   

    def kinetic_energy(system_def, q, q_dot):
        # TODO implement
        return 0.

    # ===========================================
    # === Conditional systems
    # ===========================================

    def sample_conditional_params(system_def, rngkey):
        # Sample a random, valid setting of the conditional parameters for the system.
        # (If there are no conditional parameters, returning the empty array as below is fine)
        # TODO implement
        return jnp.zeros((0,))

    # ===========================================
    # === Visualization routines
    # ===========================================
    
    def build_system_ui(system_def):
        # Construct a Polyscope gui to tweak parameters of the system
        # Make psim.InputFloat etc calls here

        # If appliciable, the cond_params values and external_forces values should be 
        # made editable in this function.

        pass

    def visualize(system_def, q):
        # Create and/or update a Polyscope visualization of the system in its current state

        # TODO implement
        pass
    

    def visualize_set_nice_view(system_def, q):
        # Set a Polyscope camera view which nicely looks at the scene
        
        # Example:
        # (could also be dependent on x)
        # ps.look_at((2., 1., 2.), (0., 0., 0.))
    
        pass
