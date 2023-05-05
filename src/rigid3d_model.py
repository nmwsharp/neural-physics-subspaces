import jax
import jax.numpy as jnp

import numpy as np

import os

import polyscope as ps
import polyscope.imgui as psim

try:
    import igl
finally:
    print("WARNING: igl bindings not available")

import utils

###

def make_body(file, density, scale):

    v, f = igl.read_triangle_mesh(file)
    v = scale*v

    vol = igl.massmatrix(v,f).data
    vol = np.nan_to_num(vol) # massmatrix returns Nans in some stewart meshes

    c = np.sum( vol[:,None]*v, axis=0 ) / np.sum(vol) 
    v = v - c

    W = np.c_[v, np.ones(v.shape[0])]
    mass = np.matmul(W.T, vol[:,None]*W) * density

    x0 = jnp.array( [[1, 0, 0],[0, 1, 0],[0, 0, 1], c] )

    body = {'v': v, 'f': f, 'W':W, 'x0': x0, 'mass': mass }
    return body

def make_joint( b0, b1, bodies, joint_pos_world, joint_vec_world ):
    # Creates a joint between the specified bodies, assumes the bodies have zero rotation and are properly aligned in the world
    # TODO: Use rotation for joint initialization
    pb0 = joint_pos_world
    vb0 = joint_vec_world
    if b0 != -1:
        c0 = bodies[b0]['x0'][3,:]
        pb0 = pb0 - c0
    pb1 = joint_pos_world
    vb1 = joint_vec_world
    if b1 != -1:
        c1 = bodies[b1]['x0'][3,:]
        pb1 = pb1 - c1
    joint = {'body_id0': b0, 'body_id1': b1, 'pos_body0': pb0, 'pos_body1': pb1, 'vec_body0': vb0, 'vec_body1': vb1}
    return joint

def bodiesToStructOfArrays(bodies):
    v_arr = []
    f_arr = []
    W_arr = []
    x0_arr = []
    mass_arr = []
    for b in bodies:
        v_arr.append(b['v'])
        f_arr.append(b['f'])
        W_arr.append(b['W'])
        x0_arr.append(b['x0'])
        mass_arr.append(b['mass'])
    
    out_struct = {
        'v'     : jnp.stack(v_arr, axis=0),
        'f'     : jnp.stack(f_arr, axis=0),
        'W'     : jnp.stack(W_arr, axis=0),
        'x0'    : jnp.stack(x0_arr, axis=0),
        'mass'  : jnp.stack(mass_arr, axis=0),
    }

    n_bodies = len(v_arr)

    return out_struct, n_bodies


class Rigid3DSystem:

    @staticmethod
    def construct(problem_name):
        system_def = {}
        system = Rigid3DSystem()

        system.system_name = "Rigid3d"
        system.problem_name = str(problem_name)

        # set some defaults
        system_def['external_forces'] = {}
        system_def['cond_param'] = jnp.zeros((0,))
        system_def["contact_stiffness"] = 1000000.0
        system.cond_dim = 0
        system.body_ID = None

        bodies = []
        joint_list = []
        numBodiesFixed = 0


        if problem_name == 'klann':

            bodies.append( make_body( os.path.join(".", "data", "klann-red.obj"), 1000, 1.0) )
            bodies.append( make_body( os.path.join(".", "data", "klann-purple.obj"), 1000, 1.0) )
            bodies.append( make_body( os.path.join(".", "data", "klann-brown.obj"), 1000, 1.0) )
            bodies.append( make_body( os.path.join(".", "data", "klann-distal.obj"), 1000, 1.0) )
            bodies.append( make_body( os.path.join(".", "data", "klann-top.obj"), 1000, 1.0) )

            joint_list.append( make_joint( 0, -1, bodies, jnp.array([ 0,           0.08      ,0.044 ]), jnp.array([ 0, 0.0, 1.0 ]) ) )
            joint_list.append( make_joint( 0,  1, bodies, jnp.array([-0.046622,    0.097594  ,0.044 ]), jnp.array([ 0, 0.0, 1.0 ]) ) ) 
            joint_list.append( make_joint( 1,  2, bodies, jnp.array([-0.1736,      0.11205   ,0.044 ]), jnp.array([ 0, 0.0, 1.0 ]) ) )
            joint_list.append( make_joint( 1,  3, bodies, jnp.array([-0.31194,     0.16654   ,0.044 ]), jnp.array([ 0, 0.0, 1.0 ]) ) )
            joint_list.append( make_joint( 4, -1, bodies, jnp.array([-0.13,        0.1875    ,0.044 ]), jnp.array([ 0, 0.0, 1.0 ]) ) )
            joint_list.append( make_joint( 2, -1, bodies, jnp.array([-0.13,        0.045     ,0.044 ]), jnp.array([ 0, 0.0, 1.0 ]) ) )
            joint_list.append( make_joint( 4,  3, bodies, jnp.array([-0.21981,     0.25102   ,0.044 ]), jnp.array([ 0, 0.0, 1.0 ]) ) )

            system_def["gravity"] = jnp.array([0.0, -0.98, 0.0])
            system_def['external_forces']['force_strength_minmax'] = (-10, 10)
            system_def['external_forces']['force_strength_x'] = 0.0
            system_def['external_forces']['force_strength_y'] = 0.0
            system_def['external_forces']['force_strength_z'] = 0.0

        elif problem_name == 'stewart':
            
            scale = 5.0 

            bodies.append( make_body( os.path.join(".", "data", "stewart-base.obj"), 1000, scale) )
            bodies.append( make_body( os.path.join(".", "data", "stewart-arm1.obj"), 1000, scale) )
            bodies.append( make_body( os.path.join(".", "data", "stewart-arm2.obj"), 1000, scale) )
            bodies.append( make_body( os.path.join(".", "data", "stewart-arm3.obj"), 1000, scale) )
            bodies.append( make_body( os.path.join(".", "data", "stewart-arm4.obj"), 1000, scale) )
            bodies.append( make_body( os.path.join(".", "data", "stewart-arm5.obj"), 1000, scale) )
            bodies.append( make_body( os.path.join(".", "data", "stewart-arm6.obj"), 1000, scale) )
            bodies.append( make_body( os.path.join(".", "data", "stewart-strut1.obj"), 1000, scale) )
            bodies.append( make_body( os.path.join(".", "data", "stewart-strut2.obj"), 1000, scale) )
            bodies.append( make_body( os.path.join(".", "data", "stewart-strut3.obj"), 1000, scale) )
            bodies.append( make_body( os.path.join(".", "data", "stewart-strut4.obj"), 1000, scale) )
            bodies.append( make_body( os.path.join(".", "data", "stewart-strut5.obj"), 1000, scale) )
            bodies.append( make_body( os.path.join(".", "data", "stewart-strut6.obj"), 1000, scale) )
            bodies.append( make_body( os.path.join(".", "data", "stewart-top.obj"), 1000, scale) )

            numBodiesFixed = 1

            ang = np.pi*2.0/3.0
            R = jnp.array([[jnp.cos(ang),  0.0, jnp.sin(ang)], [0.0,  1.0, 0.0], [-jnp.sin(ang),  0.0, jnp.cos(ang)]])
            Rh = jnp.array([[jnp.cos(ang/2),  0.0, jnp.sin(ang/2)], [0.0,  1.0, 0.0], [-jnp.sin(ang/2),  0.0, jnp.cos(ang/2)]])

            ####[x,z,-y]
            a = scale*jnp.array([-0.018,  0.0215, -0.044856])
            b = scale*jnp.array([-0.047847, 0.0215, 0.00684])
            v = jnp.array([ 0, 0.0, 1.0 ])

            Ra = jnp.matmul(R,a)
            RRa = jnp.matmul(R,Ra)
            Rb = jnp.matmul(R,b)
            RRb = jnp.matmul(R,Rb)
            Rv = jnp.matmul(R,v)
            RRv = jnp.matmul(R,Rv)

            joint_list.append( make_joint( 0,  1, bodies, a, v ) )
            joint_list.append( make_joint( 0,  2, bodies, b, Rv ) )
            joint_list.append( make_joint( 0,  3, bodies, Ra, Rv ) )
            joint_list.append( make_joint( 0,  4, bodies, Rb, RRv ) )
            joint_list.append( make_joint( 0,  5, bodies, RRa, RRv ) )
            joint_list.append( make_joint( 0,  6, bodies, RRb, v ) )

            ####[x,z,-y]
            a = scale*jnp.array([-0.003,  0.0215, -0.051856])
            b = scale*jnp.array([-0.046409, 0.0215, 0.02333])

            Ra = jnp.matmul(R,a)
            RRa = jnp.matmul(R,Ra)
            Rb = jnp.matmul(R,b)
            RRb = jnp.matmul(R,Rb)

            joint_list.append( make_joint( 1,  7, bodies, a, 0.01*v ) )
            joint_list.append( make_joint( 2,  8, bodies, b, 0.01*Rv ) )
            joint_list.append( make_joint( 3,  9, bodies, Ra, 0.01*Rv ) )
            joint_list.append( make_joint( 4,  10, bodies, Rb, 0.01*RRv ) )
            joint_list.append( make_joint( 5,  11, bodies, RRa, 0.01*RRv ) )
            joint_list.append( make_joint( 6,  12, bodies, RRb, 0.01*v ) )

            ####[x,z,-y]
            a = scale*jnp.array([-0.032159,  0.082222, -0.022686])
            b = scale*jnp.array([-0.035712, 0.082222, -0.016488 ])
            v = jnp.matmul(Rh,jnp.array([ 0, 0.0, 1.0 ]))

            Ra = jnp.matmul(R,a)
            RRa = jnp.matmul(R,Ra)
            Rb = jnp.matmul(R,b)
            RRb = jnp.matmul(R,Rb)
            Rv = jnp.matmul(R,v)
            RRv = jnp.matmul(R,Rv)

            joint_list.append( make_joint( 7,  13, bodies, a, 0.01*v ) )
            joint_list.append( make_joint( 8,  13, bodies, b, 0.01*v ) )
            joint_list.append( make_joint( 9,  13, bodies, Ra, 0.01*Rv ) )
            joint_list.append( make_joint( 10,  13, bodies, Rb, 0.01*Rv ) )
            joint_list.append( make_joint( 11,  13, bodies, RRa, 0.01*RRv ) )
            joint_list.append( make_joint( 12,  13, bodies, RRb, 0.01*RRv ) )

            ###
            system_def["gravity"] = jnp.array([0.0, 0.98, 0.0])
            system_def['external_forces']['force_strength_minmax'] = (-300, 300)
            system_def['external_forces']['force_strength_x'] = 0.0
            system_def['external_forces']['force_strength_y'] = 0.0
            system_def['external_forces']['force_strength_z'] = 0.0
        
            system.body_ID = np.array([2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2])

        else:
            raise ValueError("unrecognized system problem_name")
        
        #
        posFixed  = jnp.array( np.array([ body['x0']   for body in bodies[0:numBodiesFixed] ]).flatten() )
        pos  = jnp.array( np.array([ body['x0']   for body in bodies[numBodiesFixed:] ]).flatten() )

        mass = jnp.array( np.array([ body['mass'] for body in bodies[numBodiesFixed:] ]).flatten() )
        
        #
        system.dim = pos.size

        system.bodiesRen = bodies
        system.n_bodies = len(bodies)
        
        #
        system.joints = joint_list

        system_def['fixed_pos'] = posFixed
        system_def['rest_pos'] = pos
        system_def['init_pos'] = pos
        system_def['mass'] = mass
        system_def['dim'] = pos.size

        system_def['interesting_states'] = system_def['init_pos'][None,:]

        return system, system_def

    def potential_energy(self, system_def, q):
        
        qRFull = jnp.concatenate((system_def['fixed_pos'],q)).reshape(-1,4,3)

        ###########

        joint_energy = 0.0

        for j in self.joints:

            pb0 = j['pos_body0']
            vb0 = j['vec_body0']
            
            b0id = j['body_id0']
            if b0id != -1:
                # transform point on body to point in world
                pb0 = jnp.matmul(jnp.append(pb0,1), qRFull[b0id])
                vb0 = jnp.matmul(jnp.append(vb0,0), qRFull[b0id])

            pb1 = j['pos_body1']
            vb1 = j['vec_body1']

            b1id = j['body_id1']
            if b1id != -1:
                # transform point on body to point in world
                pb1 = jnp.matmul(jnp.append(pb1,1), qRFull[b1id])
                vb1 = jnp.matmul(jnp.append(vb1,0), qRFull[b1id])

            d = pb1 - pb0
            dist_squared = jnp.sum(d*d)
            joint_stiffness = 300000.0 

            align = 1.0-jnp.sum(vb0*vb1)
            align_stiffness = 500.0

            joint_energy += 0.5 * joint_stiffness * dist_squared + 0.5 * align_stiffness * align

        ###########

        contact_energy = 0.0
        ext_force_energy = 0.0

        external_forces = system_def['external_forces']
        forcedBodyId = 23 // 2

        if 'force_strength_x' in external_forces:
            ext_force_energy += jnp.sum(qRFull[forcedBodyId,3,0] * external_forces['force_strength_x'])

        if 'force_strength_y' in external_forces:
            ext_force_energy += jnp.sum(qRFull[forcedBodyId,3,1] * external_forces['force_strength_y'])

        if 'force_strength_z' in external_forces:
            ext_force_energy += jnp.sum(qRFull[forcedBodyId,3,2] * external_forces['force_strength_z'])

        ###########

        qR = q.reshape(-1,4,3)

        massR = system_def['mass'].reshape(-1,4,4)

        gravity = system_def["gravity"]
        c_weighted = massR[:,3,3][:,None]*qR[:,3,:] 

        gravity_energy = -jnp.sum(c_weighted * gravity[None,:])

        ###########

        rotT = qR[:,0:3,:]
        ide = jnp.stack([jnp.identity(3)]*rotT.shape[0])

        const = rotT @ jnp.swapaxes(rotT,1,2) - ide 
        rigid_energy = 5000*jnp.sum(const*const)   

        ###########

        return joint_energy + gravity_energy + ext_force_energy + rigid_energy + contact_energy

    def kinetic_energy(self, system_def, q, q_dot):

        q_dotR = q_dot.reshape(-1,4,3)
        massR = system_def['mass'].reshape(-1,4,4)
        
        A = jnp.swapaxes(q_dotR,1,2) @ massR @ q_dotR
        return 0.5*jnp.sum(jnp.trace(A, axis1=1, axis2=2))

    # ===========================================
    # === Conditional systems
    # ===========================================

    def sample_conditional_params(self, system_def, rngkey, rho=1.):
        return jnp.zeros((0,))

    def build_system_ui(self, system_def):
        if psim.TreeNode("system UI"):
            psim.TextUnformatted("External forces:")

            if "force_strength_x" in system_def["external_forces"]:
                low, high = system_def['external_forces']['force_strength_minmax']
                _, new_val = psim.SliderFloat("force_strength_x", float(system_def['external_forces'][ 'force_strength_x']), low, high)
                system_def['external_forces']['force_strength_x'] = jnp.array(new_val)

            if "force_strength_y" in system_def["external_forces"]:
                low, high = system_def['external_forces']['force_strength_minmax']
                _, new_val = psim.SliderFloat("force_strength_y", float(system_def['external_forces'][ 'force_strength_y']), low, high)
                system_def['external_forces']['force_strength_y'] = jnp.array(new_val)

            if "force_strength_z" in system_def["external_forces"]:
                low, high = system_def['external_forces']['force_strength_minmax']
                _, new_val = psim.SliderFloat("force_strength_z", float(system_def['external_forces'][ 'force_strength_z']), low, high)
                system_def['external_forces']['force_strength_z'] = jnp.array(new_val)


            psim.TreePop()

    def visualize(self, system_def, x, name="rigid3d", prefix='', transparency=1.):

        xr = jnp.concatenate((system_def['fixed_pos'],x)).reshape(-1,4,3)

        for bid in range(self.n_bodies):
            v = np.array(jnp.matmul(self.bodiesRen[bid]['W'], xr[bid]))
            f = np.array(self.bodiesRen[bid]['f'])

            ps_body = ps.register_surface_mesh("body" + prefix + str(bid), v, f)
            if transparency < 1.:
                ps_body.set_transparency(transparency)

            transform = np.identity(4)
            ps_body.set_transform( transform )
        
        return ps_body # not clear that anything needs to be returned

    def export(self, system_def, x, prefix=""):
        pass

    def visualize_set_nice_view(self, system_def, x):
        ps.look_at((1.5, 1.5, 1.5), (0., -.2, 0.))

