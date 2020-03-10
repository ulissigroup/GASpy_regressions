import gzip
import os
import fnmatch
import re
import pickle
from uuid import uuid4
from shutil import unpack_archive, rmtree
from gaspy.fireworks_helper_scripts import get_launchpad

# load regexes for fizzled error matching
errors = pickle.load(open('error_regexes.pkl', 'rb'))

class ParsefileFail(Exception):
    pass

def unzip(filename):
    """
    Unpacks an archive to '/tmp' as destination.

    Parameters:
        filename (string): full path to the file to be unpacked.

    Returns:
        dest_dir (string): path to directory containing contents of archive.
    """
    
    dest = str(uuid4())
    dest_dir = os.path.join("/tmp", dest)
    unpack_archive(filename, dest_dir)
    return dest_dir

def rm_temp(temp_dir):
    """
    Removes a directory tree.
    
    Parameters:
        temp_dir (string): full path to the directory to be removed.
        
    Returns:
        None
    """
    rmtree(temp_dir)

def readFile(filename):
    """
    Reads a text file into string
    
    Parameters:
        filename (string): full path to the file to be read.
        
    Returns:
        file_contents (string): contents of the file.
    """
    
    if filename.endswith(".gz"):
        f = gzip.open(filename, "rb")
        try:
            file_contents = f.read().decode("utf-8")
        except UnicodeDecodeError:
            file_contents = ""
        return file_contents
    else:
        with open(filename) as file:
            try:
                file_contents = file.read()
            except UnicodeDecodeError:
                file_contents = ""
            return file_contents

        
def get_avg_steptime(launch_dir):
    """
    Obtain the average time per ionic step of the given calculation.
    The result is real time in seconds, not taking into account the number of cores.
    
    Parameters:
        launch_dir (string): full path to the launch directory, containing CAR files.
        
    Returns:
        avg_steptime (float): average seconds per ionic step.
    """
    
    for file in os.listdir(launch_dir):
        if fnmatch.fnmatch(file, 'OUTCAR*'):
            filename = os.path.join(launch_dir, file)
            file_contents = readFile(filename)
            m = re.findall("LOOP\+:\s+cpu time\s*\d*\.\d*:\s*real time\s*(\d*\.\d*)", file_contents)
            if m:
                m = list(map(lambda x: float(x), m))
                avg_steptime = sum(m) / len(m)
                return avg_steptime
            else:
                raise ParsefileFail
                

def get_n_cores(launch_dir):
    """
    Obtain the number of cores used to run the calculation.
    
    Parameters:
        launch_dir (string): full path to the launch directory, containing CAR files.
        
    Returns:
        num_cores (int): number of cores used in calculation
    """
    
    for file in os.listdir(launch_dir):
        if fnmatch.fnmatch(file, 'OUTCAR*'):
            filename = os.path.join(launch_dir, file)
            file_contents = readFile(filename)
            m = re.search("(\d+) total cores", file_contents)
            if m:
                n_cores = int(m.group(1))
                return n_cores
            else:
                raise ParsefileFail
                
def get_steptime_coresec(launch_dir):
    """
    Obtain the average runtime per ionic step in core-seconds.
    
    Parameters:
        launch_dir (string): full path to the launch directory, containing CAR files.
        
    Returns:
        steptime_coresec (float): average time per ionic step in core-seconds.
    """
    
    avg_steptime = get_avg_steptime(launch_dir)
    n_cores = get_n_cores(launch_dir)
    steptime_coresec = avg_steptime * n_cores
    return steptime_coresec

def get_fizzled_reason(launch_dir):
    """
    Obtain the reason for fizzle by regex matching.
    
    Parameters:
        launch_dir (string): full path to the launch directory, containing an error file.
        
    Returns:
        fizzled_reason (string): a regex string that matched with the error file.
    """
    
    for file in os.listdir(launch_dir):
        if fnmatch.fnmatch(file, '*error*'):
            filename = os.path.join(launch_dir, file)
            file_contents = readFile(filename)
            for regex in errors:
                match = re.search(regex, file_contents)
                if match:
                    fizzled_reason = regex
                    return fizzled_reason

def get_state_by_launch_id(launch_id):
    """
    Obtain the state, or job status, of the firework by the given launch id
    
    Parameters:
        launch_id (int): launch id of the firework
        
    Returns:
        state (string): job status (e.g. COMPLETED, FIZZLED, READY, DEFUSED)
    """
    
    lpad = get_launchpad()
    launch_fw = list(lpad.launches.find({"launch_id" : launch_id}))[0]
    fw_id = launch_fw["fw_id"]
    fw = lpad.get_fw_by_id(fw_id)
    state = fw.state
    return state