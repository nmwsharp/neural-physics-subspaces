import jax
import jax.numpy as jnp
from jax import grad, jit, vmap

import numpy as np
import potpourri3d as pp3d

import os
from io import StringIO

import polyscope as ps
import polyscope.imgui as psim

import system_utils
import utils

import igl

def linear_energy(system_def, FT, mesh):

    dim = mesh["Vrest"].shape[1]
    poisson = system_def['poisson']
    Y = system_def['Y']
    A = mesh["A"]

    mu = 0.5 * Y / (1.0 + poisson)
    lamb = (Y * poisson) / ((1.0 + poisson)*(1.0 - 2.0*poisson)) # Plane strain condition (thick)

    E = 0.5*(FT + jnp.swapaxes(FT,1,2)) - jnp.eye(dim)[None,:,:]
    energies = mu*(E * E).sum(axis=(1, 2)) + 0.5*lamb*E.trace(axis1=1, axis2=2)**2

    return (A*energies).sum()
    
def StVK_energy(system_def, FT, mesh):

    dim = mesh["Vrest"].shape[1]
    poisson = system_def['poisson']
    Y = system_def['Y']
    A = mesh["A"]

    mu = 0.5 * Y / (1.0 + poisson)
    lamb = (Y * poisson) / ((1.0 + poisson)*(1.0 - 2.0*poisson)) # Plane strain condition (thick)
    # lamb = (Y * poisson) / (1.0 - poisson*poisson) # Plane stress condition (thin)

    E = 0.5*(FT @ jnp.swapaxes(FT,1,2) - jnp.eye(dim)[None,:,:])
    energies = mu*(E * E).sum(axis=(1, 2)) + 0.5*lamb*E.trace(axis1=1, axis2=2)**2

    return (A*energies).sum()

def neohook_energy(system_def, FT, mesh):

    poisson = system_def['poisson']
    Y = system_def['Y']
    A = mesh["A"]

    mu = 0.5 * Y / (1.0 + poisson)
    lamb = (Y * poisson) / ((1.0 + poisson)*(1.0 - 2.0*poisson)) # Plane strain condition (thick)
    # lamb = (Y * poisson) / (1.0 - poisson*poisson) # Plane stress condition (thin)


    dim = mesh["Vrest"].shape[1]
    lambRep = lamb + mu
    alpha = 1.0 + (mu / lambRep)

    FTn = (FT * FT).sum(axis=(1, 2))
    dec = jnp.linalg.det(FT) - alpha

    energies0 = 0.5*( mu*dim + lambRep*(1.0-alpha)*(1.0-alpha) )
    energies = 0.5*( mu*FTn + lambRep*dec*dec )

    return (A*(energies - energies0)).sum()


def neohook_thin_energy(system_def, FT, mesh):

    poisson = system_def['poisson']
    Y = system_def['Y']
    A = mesh["A"]

    mu = 0.5 * Y / (1.0 + poisson)
    # lamb = (Y * poisson) / ((1.0 + poisson)*(1.0 - 2.0*poisson)) # Plane strain condition (thick)
    lamb = (Y * poisson) / (1.0 - poisson*poisson) # Plane stress condition (thin)


    dim = mesh["Vrest"].shape[1]
    lambRep = lamb + mu
    alpha = 1.0 + (mu / lambRep)

    FTn = (FT * FT).sum(axis=(1, 2))
    dec = jnp.linalg.det(FT) - alpha

    energies0 = 0.5*( mu*dim + lambRep*(1.0-alpha)*(1.0-alpha) )
    energies = 0.5*( mu*FTn + lambRep*dec*dec )

    return (A*(energies - energies0)).sum()

def fem_energy(system_def, mesh, material_energy, V):

    E = mesh["E"]
    DTI = mesh["DTI"]

    DT = V[E[:, 1:]] - V[E[:, 0]][:,None]
    FT = DTI @ DT

    return material_energy(system_def, FT, mesh)

def mean_strain_metric(system_def, mesh, V):

    dim = mesh["Vrest"].shape[1]

    E = mesh["E"]
    DTI = mesh["DTI"]
    A = mesh["A"]

    DT = V[E[:, 1:]] - V[E[:, 0]][:,None]
    FT = DTI @ DT

    E = 0.5*(FT @ jnp.swapaxes(FT,1,2) - jnp.eye(dim)[None,:,:])

    rigidity_density = jnp.sqrt((E * E).sum(axis=(1, 2)))

    return (A*rigidity_density).sum() / A.sum()

def tet_mesh_boundary_faces(tets):
    # numpy to numpy
    return igl.boundary_facets(tets)

def precompute_mesh(mesh):
   
    dim = mesh["Vrest"].shape[1]
    Vrest = mesh["Vrest"]
    E = mesh["E"]

    DT = Vrest[E[:, 1:]] - Vrest[E[:, 0]][:,None]
    DTI = jnp.linalg.inv(DT)
    A = jnp.abs(jnp.linalg.det(DT)) / (dim * (dim-1))
    
    mesh["DTI"] = DTI
    mesh["A"] = A

    VA = jnp.zeros(Vrest.shape[0])
    Atile = jnp.tile(A, dim + 1).reshape((A.shape[0], dim + 1)) / (dim + 1)
    VA = VA.at[E].add(Atile)

    mesh["VA"] = VA

    if E.shape[-1] == 4:
        mesh['boundary_triangles'] = tet_mesh_boundary_faces(E)

    return mesh

def build_quad_mesh():

    mesh = {}

    mesh["Vrest"] = jnp.array([
                              [0., 0.],
                              [1., 0.],
                              [0., 1.],
                              [1., 1.]
                              ])
    mesh["E"] = jnp.array([
                          [0,1,2],
                          [1,3,2],
                          ])
    return mesh

def build_tet_mesh():

    mesh = {}

    p = np.array([1.0, 0., 0.])

    ang = np.pi*2.0/3.0
    R = np.array([[np.cos(ang),  0.0, np.sin(ang)], [0.0,  1.0, 0.0], [-np.sin(ang),  0.0, np.cos(ang)]])

    Rp = np.matmul(R, p)
    RRp = np.matmul(R, Rp)

    mesh["Vrest"] = np.array([
                              [0., 1.2, 0.],
                              p,
                              Rp,
                              RRp
                              ])
    mesh["E"] = np.array([
                          [0,1,2,3]
                          ])

    return mesh

def load_obj(filename):

    verts, faces = pp3d.read_mesh(filename)
    
    mesh = {}
    mesh["Vrest"] = verts[:,:2]
    mesh["E"] = faces

    return mesh

def load_tri_mesh( file_name_root ):

    # ele file
    lines = open(file_name_root+".ele").readlines()
    lines = [line for line in lines if line.strip() != '' and line[0] != '#']
    n_tri, n_nodesPerTriangle, n_attribT = map(int, lines[0].split())
    faces = np.loadtxt(StringIO(''.join(lines[1:])), dtype=int)[:,1:4]

    regions = 0
    if n_attribT > 0:
        regions = np.loadtxt(StringIO(''.join(lines[1:])), dtype=int)[:,4]

    # node file
    lines = open(file_name_root+".node").readlines()
    lines = [line for line in lines if line.strip() != '' and line[0] != '#']
    n_vert, n_dim, n_attrib, n_bmark = map(int, lines[0].split())
    verts = np.loadtxt(StringIO(''.join(lines[1:])), dtype=float)[:,1:3]

    center = (verts.max(axis = 0) + verts.min(axis = 0))/2
    scale = verts.max(axis = 0) - verts.min(axis = 0)
    verts = (verts - center)/scale.max() 

    first_node_index = np.loadtxt(StringIO(''.join(lines[1])), dtype=int)[0]
    faces = faces - first_node_index

    #
    mesh = {}
    mesh["Vrest"] = verts
    mesh["E"] = faces
    if n_attribT > 0:
        Y = np.full( (1, n_tri), 1.5e3 )
        np.put( Y, np.argwhere(regions[:] == 2), 1.5e2 )
        poisson = np.full( (1, n_tri), 0.4 )
        np.put( poisson, np.argwhere(regions[:] == 2), 0.4 )
        mesh["Y"] = Y
        mesh["poisson"] = poisson
        mesh["regions"] = regions

    return mesh

def load_tet_mesh_igl(file_name_root, normalize=True):
    verts, tets, _ = igl.read_mesh( file_name_root )

    if normalize:
        center = (verts.max(axis = 0) + verts.min(axis = 0))/2
        scale = verts.max(axis = 0) - verts.min(axis = 0)
        verts = (verts - center)/scale.max()

    mesh = {}
    mesh["Vrest"] = verts
    mesh["E"] = tets

    return mesh


def load_tet_mesh( file_name_root ):

    # ele file
    lines = open(file_name_root+".ele").readlines()
    lines = [line for line in lines if line.strip() != '' and line[0] != '#']
    n_tri, n_nodesPerTriangle, n_attribT = map(int, lines[0].split())
    data = np.loadtxt(StringIO(''.join(lines[1:])), dtype=int)[:,1:]
    tets = data[:,0:4]
    regions = 0
    if n_attribT > 0:
        regions = data[:,-1]

    # node file
    lines = open(file_name_root+".node").readlines()
    lines = [line for line in lines if line.strip() != '' and line[0] != '#']
    n_vert, n_dim, n_attrib, n_bmark = map(int, lines[0].split())
    verts = np.loadtxt(StringIO(''.join(lines[1:])), dtype=float)[:,1:4]

    center = (verts.max(axis = 0) + verts.min(axis = 0))/2
    scale = verts.max(axis = 0) - verts.min(axis = 0)
    verts = (verts - center)/scale.max()

    first_node_index = np.loadtxt(StringIO(''.join(lines[1])), dtype=int)[0]
    tets = tets - first_node_index

    mesh = {}
    mesh["Vrest"] = verts
    mesh["E"] = tets
    if n_attribT > 0:
        Y = np.full((1, n_tri), 100e2)
        np.put(Y, np.argwhere(regions[:] == 0), 1e2)
        poisson = np.full((1, n_tri), 0.4)
        np.put(poisson, np.argwhere(regions[:] == 0), 0.4)
        mesh["Y"] = Y
        mesh["poisson"] = poisson
        mesh["regions"] = regions

    return mesh


###

class FEMSystem:

    def __init__(self):
        
        self.mesh = None

    @staticmethod
    def construct(problem_name):

        system_def = {}
        system = FEMSystem()

        system.system_name = "FEM"
        system.problem_name = str(problem_name)

        # set some defaults
        system_def['external_forces'] = {}
        system_def['cond_param'] = jnp.zeros((0,))
        system.cond_dim = 0
       
        def get_full_position(self, system_def, q):
            pos = system_utils.apply_fixed_entries( 
                    system_def['fixed_inds'], system_def['unfixed_inds'], 
                    system_def['fixed_values'], q).reshape(-1, self.pos_dim)
            return pos
        system.get_full_position = get_full_position
        
        def update_conditional(self, system_def):
            return system_def # default does nothing
        system.update_conditional = update_conditional
            
        system.material_energy = neohook_energy
    


        if problem_name == 'bistable':

            mesh = load_tri_mesh( os.path.join(".", "data", "longerCantileverP2" ) )
            mesh["Vrest"][:,1] = 0.5 * mesh["Vrest"][:,1]

            mesh = precompute_mesh(mesh)
            
            # system_def["gravity"] = jnp.array([0., -0.98])
            system_def["gravity"] = jnp.array([0., 0.])
            system_def['poisson'] = jnp.array(0.45)
            system_def['Y'] = jnp.array(1e3)
            system_def['density'] = jnp.array(10.0)

            verts = mesh["Vrest"]

            verts_compress = verts
            verts_compress[:,0] *= 0.8
            verts_init = verts*0.8
            
            verts = jnp.array(verts)
            verts_compress = jnp.array(verts_compress)
            verts_init = jnp.array(verts_init)

            # identify verts that are on the x min and pin them.
            xmin = jnp.amin(verts, axis = 0)[0]
            xmax = jnp.amax(verts, axis = 0)[0]
            pinned_verts_mask = jnp.logical_or(verts[:,0] < (xmin + 1e-3),  verts[:,0] > xmax - 1e-3)
            pinned_verts_mask_flat = jnp.repeat(pinned_verts_mask,2)
            fixed_inds, unfixed_inds, fixed_values, unfixed_values = \
                system_utils.generate_fixed_entry_data(pinned_verts_mask_flat, verts_compress.flatten())
            system_def["fixed_inds"] = fixed_inds
            system_def["unfixed_inds"] = unfixed_inds
            system_def["fixed_values"] = fixed_values

            # configure external forces
            xmid = (xmin + xmax) / 2;
            system_def['external_forces']['force_verts_mask'] = jnp.logical_and((verts[:,0] > xmid - 1e-1), (verts[:,0] < xmid + 1e-1))
            system_def['external_forces']['pull_X'] = jnp.array(0.)
            system_def['external_forces']['pull_Y'] = jnp.array(0.)
            pull_minmax = (-0.1, 0.1)
            system_def['external_forces']['pull_strength_minmax'] = pull_minmax
            system_def['external_forces']['pull_strength'] = 0.5 * (pull_minmax[0] + pull_minmax[1])

            system.mesh = mesh
            system_def['init_pos'] = unfixed_values
            system.pos_dim = verts.shape[1]
            system.dim = system_def['init_pos'].size
        

        elif problem_name == 'load3d':

            mesh = load_tet_mesh_igl( os.path.join(".", "data", "beam365.mesh" ) )
            mesh = precompute_mesh(mesh)
            
            system.material_energy = StVK_energy
            
            system_def["gravity"] = jnp.array([0, -0.98, 0])
            system_def['poisson'] = jnp.array(0.45)
            system_def['Y'] = jnp.array(5e3)
            system_def['density'] = jnp.array(100.0)

            verts = jnp.array(mesh["Vrest"])

            # identify verts that are on the x min and pin them.
            xmin = jnp.amin( verts, axis = 0 )[2]
            pinned_verts_mask = verts[:,2] < xmin + 1e-3
            pinned_verts_mask_flat = jnp.repeat(pinned_verts_mask,3)
            fixed_inds, unfixed_inds, fixed_values, unfixed_values = \
                system_utils.generate_fixed_entry_data(pinned_verts_mask_flat, verts.flatten())
            system_def["fixed_inds"] = fixed_inds
            system_def["unfixed_inds"] = unfixed_inds
            system_def["fixed_values"] = fixed_values

            xmax = np.amax( verts, axis = 0 )[2]
            system_def['force_verts_mask'] = verts[:,2] > (xmax - 1e-3)

            system.mesh = mesh
            system_def['init_pos'] = unfixed_values
            system.pos_dim = verts.shape[1]
            system.dim = system_def['init_pos'].size
        
            
        elif problem_name.startswith('heterobeam'):

            mesh = load_tet_mesh( os.path.join(".", "data", "heterobeam" ) )
            mesh = precompute_mesh(mesh)
            
            if problem_name == 'heterobeam-gravity':
                system_def["gravity"] = jnp.array([0, -1.0, 0])
            else:
                system_def["gravity"] = jnp.array([0, -1, 0])
        
            system_def['poisson'] = jnp.array(mesh["poisson"])
            system_def['Y'] = jnp.array(mesh["Y"])
            system_def['density'] = jnp.array(100.0)

            verts = jnp.array(mesh["Vrest"])

            # identify verts that are on the x min and pin them.
            xmin = jnp.amin( verts, axis = 0 )[2]
            pinned_verts_mask = verts[:,2] < xmin + 1e-3
            pinned_verts_mask_flat = jnp.repeat(pinned_verts_mask,3)
            fixed_inds, unfixed_inds, fixed_values, unfixed_values = \
                system_utils.generate_fixed_entry_data(pinned_verts_mask_flat, verts.flatten())
            system_def["fixed_inds"] = fixed_inds
            system_def["unfixed_inds"] = unfixed_inds
            system_def["fixed_values"] = fixed_values

            xmax = np.amax( verts, axis = 0 )[2]
   
            # configure external forces
            xmax = np.amax( verts, axis = 0 )[2]
            system_def['external_forces']['force_verts_mask'] = verts[:,2] > xmax - 1e-3
            system_def['external_forces']['pull_X'] = jnp.array(0.)
            system_def['external_forces']['pull_Y'] = jnp.array(0.)
            system_def['external_forces']['pull_Z'] = jnp.array(0.)
            pull_minmax = (-0.005, 0.005)
            system_def['external_forces']['pull_strength_minmax'] = pull_minmax
            system_def['external_forces']['pull_strength'] = 0.5 * (pull_minmax[0] + pull_minmax[1])

            system.mesh = mesh
            system_def['init_pos'] = unfixed_values
            system.pos_dim = verts.shape[1]
            system.dim = system_def['init_pos'].size

        else:
            raise ValueError("unrecognized system problem_name")
        
        system_def['interesting_states'] = system_def['init_pos'][None,:]

        return system, system_def


    # ===========================================
    # === Energy functions 
    # ===========================================

    def mean_strain(self, system_def, q):

        pos = self.get_full_position(self, system_def, q)

        return mean_strain_metric(system_def, self.mesh, pos)

    def potential_energy(self, system_def, q):
        system_def = self.update_conditional(self, system_def)

        pos = self.get_full_position(self, system_def, q)

        mass_lumped = self.mesh["VA"] * system_def['density']

        contact_energy = 0


        gravity = system_def["gravity"]
        gravity_energy = -jnp.sum(pos * gravity[None,] * mass_lumped[:,None])

        ext_force_energy = jnp.array(0.)
        if 'pull_X' in system_def['external_forces']:
            mask = system_def['external_forces']["force_verts_mask"]
            if pos.shape[1] == 2:
                dir_force = jnp.array([[1., 0.]])
            else:
                dir_force = jnp.array([[1., 0., 0.]])
            masked_force = mask[:,None] * dir_force * system_def['external_forces']['pull_X'] * system_def['external_forces']['pull_strength'] 
            ext_force_energy += jnp.sum(masked_force*pos)

        if 'pull_Y' in system_def['external_forces']:
            mask = system_def['external_forces']["force_verts_mask"]
            if pos.shape[1] == 2:
                dir_force = jnp.array([[0., 1.]])
            else:
                dir_force = jnp.array([[0., 1., 0.]])
            masked_force = mask[:,None] * dir_force * system_def['external_forces']['pull_Y'] * system_def['external_forces']['pull_strength'] 
            ext_force_energy += jnp.sum(masked_force*pos)

        if 'pull_Z' in system_def['external_forces']:
            mask = system_def['external_forces']["force_verts_mask"]
            if pos.shape[1] == 2:
                dir_force = jnp.array([[0., 0.]])
            else:
                dir_force = jnp.array([[0., 0., 1.]])
            masked_force = mask[:,None] * dir_force * system_def['external_forces']['pull_Z'] * system_def['external_forces']['pull_strength'] 
            ext_force_energy += jnp.sum(masked_force*pos)

        return fem_energy(system_def, self.mesh, self.material_energy, pos) + gravity_energy + contact_energy + ext_force_energy
   
    def kinetic_energy(self, system_def, q, q_dot):
        system_def = self.update_conditional(self, system_def)

        pos_dot = system_utils.apply_fixed_entries(
                    system_def['fixed_inds'], system_def['unfixed_inds'], 
                    0., q_dot).reshape(-1, self.pos_dim)

        mass_lumped = self.mesh["VA"] * system_def['density']

        return 0.5 * jnp.sum(mass_lumped * jnp.sum(jnp.square(pos_dot), axis=-1))
    
    # ===========================================
    # === Conditional systems
    # ===========================================

    def sample_conditional_params(self, system_def, rngkey, rho=1.):
        return jnp.zeros((0,))

    # ===========================================
    # === Visualization routines
    # ===========================================


    def build_system_ui(self, system_def):

        if psim.TreeNode("system UI"):

            psim.TextUnformatted("External forces:")

            if "pull_X" in system_def["external_forces"]:
                pulling = system_def['external_forces']['pull_X']
                _, pulling = psim.Checkbox("pull_X", pulling)
                system_def['external_forces']['pull_X'] = jnp.where(pulling, 1., 0.)

            if "pull_Y" in system_def["external_forces"]:
                pulling = system_def['external_forces']['pull_Y']
                _, pulling = psim.Checkbox("pull_Y", pulling)
                system_def['external_forces']['pull_Y'] = jnp.where(pulling, 1., 0.)

            if "pull_Z" in system_def["external_forces"]:
                pulling = system_def['external_forces']['pull_Z']
                _, pulling = psim.Checkbox("pull_Z", pulling)
                system_def['external_forces']['pull_Z'] = jnp.where(pulling, 1., 0.)
            
            if "pull_strength" in system_def["external_forces"]:
                low, high = system_def['external_forces']['pull_strength_minmax']
                _, system_def['external_forces']['pull_strength'] = psim.SliderFloat("pull_strength", system_def['external_forces']['pull_strength'], low, high)
            
            psim.TreePop()

    def visualize(self, system_def, q, prefix="", transparency=1.0):
        system_def = self.update_conditional(self, system_def)

        name = self.problem_name + prefix

        pos = self.get_full_position(self, system_def, q)

        elem_list = self.mesh['E']

        if self.pos_dim == 2:
            ps_elems = ps.register_surface_mesh(name + " mesh", pos, np.array(elem_list))
            if 'regions' in self.mesh.keys():
                regions = self.mesh['regions']
                ps_elems.add_scalar_quantity("material colors r", regions, defined_on='faces', enabled=True)
        else:
            if 'regions' in self.mesh.keys():
                ps_elems = ps.register_volume_mesh(name + " mesh", pos, np.array(elem_list))
                regions = self.mesh['regions']
                ps_elems.add_scalar_quantity("material colors r", regions, defined_on='cells', enabled=True)
            else:
                ps_elems = ps.register_surface_mesh(name + " mesh", pos, np.array(self.mesh['boundary_triangles']))
            
        if(transparency < 1.):
            ps_elems.set_transparency(transparency)


        if self.problem_name == "bistable":
            s = 0.4
            w = 0.07
            h = 0.1
            quad_block_verts = np.array([
                [-s-w,-h],
                [-s  ,-h],
                [-s  ,+h],
                [-s-w,+h],
                [+s  ,-h],
                [+s+w,-h],
                [+s+w,+h],
                [+s  ,+h],
                ])
            ps_endcaps = ps.register_surface_mesh("endcaps", quad_block_verts, np.array([[0,1,2,3], [4,5,6,7]]), color=(0.7,0.7,0.7), edge_width=4.)
        
        
        return (ps_elems)

    def export(self, system_def, x, prefix=""):

        system_def = self.update_conditional(self, system_def)

        pos = self.get_full_position(self, system_def, x)
        tri_list = self.mesh['boundary_triangles']

        filename = prefix+f"{self.problem_name}_mesh.obj"

        utils.write_obj(filename, pos, np.array(tri_list))

    def visualize_set_nice_view(self, system_def, q):

        if self.problem_name == 'spot':
            ps.look_at((-1.2, 0.8, -1.8), (0., 0., 0.))

        ps.look_at((2., 1., 2.), (0., 0., 0.))
