import random
import warnings
from gaspy import gasdb
from tqdm import tqdm_notebook
from gaspy.fireworks_helper_scripts import (get_launchpad,
                                            get_atoms_from_fwid,
                                            decode_trajhex_to_atoms)


def get_initial_atoms_from_fwid(fwid):
    '''
    Function written by Kevin
    This function will return the `ase.Atoms` object given to a rocket.

    Args:
        fwid    Integer indicating the FireWorks ID that you're trying to get
                the atoms object for
    Returns
        atoms   `ase.Atoms` instance from the Firework you provided
    '''
    # Get the firework object, which will have all the information we'll need
    try:
        lpad = get_launchpad()
        fw = lpad.get_fw_by_id(fwid)
    # Close the Mongo connection for cleanliness' sake
    finally:
        lpad.fireworks.database.client.close()

    # Get the Firework task that was meant to convert the original hexstring to
    # a trajectory file. We'll get the original atoms from this task (in
    # hexstring format). Note that over the course of our use, we have had
    # different names for these FireWorks tasks, so we check for them all.
    function_names_of_hex_encoders = set(['vasp_functions.hex_to_file',
                                          'fireworks_helper_scripts.atoms_hex_to_file',
                                          'fireworks_helper_scripts.atomsHexToFile'])
    trajhexes = [task['args'][1] for task in fw.spec['_tasks']
                 if task.get('func', '') in function_names_of_hex_encoders]
    # If there was not one task, then ignore it because it's probably some old
    # firework that isn't formatted correctly for us
    if len(trajhexes) != 1:
        warnings.warn('%i does not have any trajhexes. Moving on.'% fwid, RuntimeWarning)
        return None
    # If the trajhex is empty, then it probably bugged out. Ignore it.
    elif not trajhexes[0]:
        warnings.warn('%i has an empty trajhex. Moving on.'% fwid, RuntimeWarning)
        return None

    # Use our vanilla trajhex opener
    try:
        atoms = decode_trajhex_to_atoms(trajhexes[0])
    # Some hexes are weird and don't open properly for a bunch of reasons. Ignore them.
    except:
        warnings.warn('%i did not open properly. Moving on.'% fwid, RuntimeWarning)
        return None
    return atoms