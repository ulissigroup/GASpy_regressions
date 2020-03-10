import spglib
import pickle
import numpy as np
from gaspy.atoms_operators import fingerprint_adslab
from gaspy.mongo import make_spglib_cell_from_atoms

# load in number of electrons for each atom from pseudopotentials
big_numbers_dict = pickle.load(open("big_numbers_dict.pkl", "rb"))
big_valency_dict = pickle.load(open("big_valency_dict.pkl", "rb"))


# since we can not get vasp settings from atoms alone, we will default to PBE 5.4
pp = "PBE"
pp_version = "5.4"


def get_cell_volume(atoms):
    """
    Get the unit cell volume of atoms object. 
    
    Parameters:
        atoms (ase.Atoms): atoms object to create spglib cell from.
        
    Returns:
        volume (float): volume of unit cell.
    """
    
    volume = atoms.get_volume()
    return volume


def get_primitive_in_unit_cell(atoms, primitive):
    """
    Get the number of primitive cells that fit into the unit cell volume.
    unit cell volume divided by primitive cell volume.
    
    Parameters:
        atoms (ase.Atoms): atoms object.
        primitive (tuple): spglib cell for primitive cell.
        
    Returns:
        n_cells (float): number of primitive cells in unit cell.
    """
    
    primitive_volume = np.abs(np.linalg.det(primitive[0]))
    
    volume = get_cell_volume(atoms)
    n_cells = volume/primitive_volume
    return n_cells


def get_surface_area(atoms):
    """
    Get the surface area of the given atoms object. 
    
    Parameters:
        atoms (ase.Atoms): atoms object.
        
    Returns:
        surface_area (float): surface area of unit cell.
    """
    
    cell = make_spglib_cell_from_atoms(atoms)
    surface_area = np.linalg.norm(np.cross(cell[0][0], cell[0][1]), ord=1)
    return surface_area


def get_n_atoms(atoms):
    """
    Get the number of atoms in the given atoms object. 
    
    Parameters:
        atoms (ase.Atoms): atoms object.
        
    Returns:
        n_atoms (int): number of atoms in the unit cell.
    """
    
    n_atoms = len(atoms.get_atomic_numbers())
    return n_atoms


def get_n_electrons(atoms):
    """
    Get the number of electrons in the given atoms object. 
    
    Parameters:
        atoms (ase.Atoms): atoms object.
        
    Returns:
        n_electrons (int): number of electrons in the unit cell.
    """
    
    pp_dir = "%s/potpaw_%s"%(pp_version, pp)
    atomic_numbers_list = atoms.get_atomic_numbers()
    n_electrons = 0
    for number in atomic_numbers_list:
        n_electrons += big_numbers_dict[pp_dir][number]
    return n_electrons


def get_primitive_cell(atoms):
    """
    Get the primitive cell for spglib given an atoms object. 
    
    Parameters:
        atoms (ase.Atoms): atoms object.
        
    Returns:
        primitive (cell): spglib tuple for primitive cell.
    """
    
    cell = make_spglib_cell_from_atoms(atoms)
    primitive = spglib.find_primitive(cell, symprec = 1e-10)
    if primitive is None:
        return cell
    return primitive


def get_n_atoms_primitive(primitive):
    """
    Get the number of atoms inside the primitive cell.
    
    Parameters:
        primitive (tuple): spglib tuple for primitive cell.
        
    Returns:
        n_atoms_primitive (int): number of atoms in primitive cell.
    """
    
    n_atoms_primitive = len(primitive[2])
    return n_atoms_primitive


def get_n_electrons_primitive(primitive):
    """
    Get the number of electrons inside the primitive cell.
    
    Parameters:
        primitive (tuple): spglib tuple for primitive cell.
        
    Returns:
        n_electrons_primitive (int): number of electrons in primitive cell.
    """

    pp_dir = "%s/potpaw_%s"%(pp_version, pp)
    n_electrons_primitive = 0
    for number in primitive[2]:
        n_electrons_primitive += big_numbers_dict[pp_dir][number]
    return n_electrons_primitive
    
    
def get_n_elems(atoms):
    """
    Get the number of different elements in the atoms object.
    
    Parameters:
        atoms (ase.Atoms): atoms object.
    
    Returns:
        n_elems (int): number of different elements.
    """
    
    n_elems = len(set(atoms.get_atomic_numbers()))
    return n_elems

def get_n_atoms_nextnearestcoordination(atoms):
    """
    Get the number of atoms from the 'next nearest coordination'.
    
    Parameters:
        atoms (ase.Atoms): atoms object.
        
    Returns:
        n_atoms_nextnearestcoordination (int): number of atoms from 'next nearest coordination'.
    """
    
    fp_init = fingerprint_adslab(atoms)
    nextnearestcoordination = fp_init["nextnearestcoordination"]
    nextnearestcoordination_atoms = nextnearestcoordination.split("-")
    nextnearestcoordination_atoms = list(filter(lambda s: not (s.isspace() or s == ""), nextnearestcoordination_atoms))
    n_atoms_nextnearestcoordination = len(nextnearestcoordination_atoms)
    return n_atoms_nextnearestcoordination

def get_n_electrons_nextnearestcoordination(atoms):
    """
    Get the number of electrons from the 'next nearest coordination'.
    
    Parameters:
        atoms (ase.Atoms): atoms object.
        
    Returns:
        n_electrons_nextnearestcoordination (int): number of electrons from 'next nearest coordination'.
    """

    fp_init = fingerprint_adslab(atoms)
    nextnearestcoordination = fp_init["nextnearestcoordination"]
    nextnearestcoordination_atoms = nextnearestcoordination.split("-")
    nextnearestcoordination_atoms = list(filter(lambda s: not (s.isspace() or s == ""), nextnearestcoordination_atoms))
    
    pp_dir = "%s/potpaw_%s"%(pp_version, pp)
    n_electrons_nextnearestcoordination = 0
    for atom in nextnearestcoordination_atoms:
        n_electrons_nextnearestcoordination += big_valency_dict[pp_dir][atom]
    return n_electrons_nextnearestcoordination

def get_features(atoms):
    """
    Get training features from a given atoms object.
    
    Parameters:
        atoms (ase.Atoms): atoms object.
        
    Returns:
        features (list): 10 features in order 
            [cell volume, 
             number of primitive cells in unit cell,
             surface area,
             number of atoms
             number of electrons,
             number of atoms in primitive cell,
             number of electrons in primitive cell,
             number of different elements,
             number of atoms in nextnearestcoordination
             number of electrons in nextnearestcoordination]
    """
    primitive = get_primitive_cell(atoms)
    cell_volume = get_cell_volume(atoms)
    n_cells = get_primitive_in_unit_cell(atoms, primitive)
    surface_area = get_surface_area(atoms)
    n_atoms = get_n_atoms(atoms)
    n_electrons = get_n_electrons(atoms)
    n_atoms_primitive = get_n_atoms_primitive(primitive)
    n_electrons_primitive = get_n_electrons_primitive(primitive)
    n_elems = get_n_elems(atoms)
    n_atoms_nextnearestcoordination = get_n_atoms_nextnearestcoordination(atoms)
    n_electrons_nextnearestcoordination = get_n_electrons_nextnearestcoordination(atoms)
    
    return [cell_volume, 
            n_cells,
            surface_area,
            n_atoms,
            n_electrons,
            n_atoms_primitive,
            n_electrons_primitive,
            n_elems,
            n_atoms_nextnearestcoordination,
            n_electrons_nextnearestcoordination]