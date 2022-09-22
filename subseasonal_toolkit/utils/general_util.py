# General utility functions for all parts of the pipeline
import os
import shutil as sh
import sys
import time
import warnings
import subprocess
import datetime
from ttictoc import Timer

DATETIME_FORMAT = '%Y%m%d'

def printf(str):
    """Calls print on given argument and then flushes
    stdout buffer to ensure printed message is displayed right away
    """
    print(str, flush=True)

# Define tic and toc functions
_TICTOC_HELPER_CLASS_5da0381c_27af_4d67_8881 = Timer(matlab_like=True)
tic = _TICTOC_HELPER_CLASS_5da0381c_27af_4d67_8881.start
def toc():
    printf(f"elapsed: {_TICTOC_HELPER_CLASS_5da0381c_27af_4d67_8881.stop()}s")
    

def make_directories(dirname):
    """Creates directory and parent directories with 777 permissions
    if they do not exist
    """
    if dirname != '':
        os.umask(0)
        os.makedirs(dirname, exist_ok=True, mode=0o777)


def make_parent_directories(file_path):
    """Creates parent directories of a given file with 777 permissions
    if they do not exist
    """
    make_directories(os.path.dirname(file_path))


def symlink(src, dest, use_abs_path=False):
    """Symlinks dest to point to src; if dest was previously a symlink,
    unlinks src first

    Args:
      src - source file name
      dest - target file name
      use_abs_path - if True, links dest to the absolute path of src
        (useful when src is not expressed relative to dest)
    """
    # n and f flags ensure that prior symlink is overwritten by new one
    if use_abs_path:
        src = os.path.abspath(src)
    cmd = "ln -nsf {} {}".format(src, dest)
    subprocess.call(cmd, shell=True)


def set_file_permissions(file_path, skip_if_exists=False, throw=False,
                         mode=0o777):
    """Set file/folder permissions.

    Parameters
    ----------
    skip_if_exists : boolean
        If True, skips setting permissions if file exists

    throw : boolean
        If True, throws exception if cannot set permissions

    """
    if not skip_if_exists or (skip_if_exists and not os.path.exists(file_path)):
        # Set permissions
        try:
            os.chmod(file_path, mode)
            sh.chown(file_path, group='sched_mit_hill')
        except Exception as err:
            if throw:
                raise err
            else:
                pass


def get_task_from_string(task_str):
    """
    Gets a region, gt_id, horizon from a task string. Returns None if invalid 
    task string
    Args:
        task_str: string in format "<region>_<gt_id>_<horzion>
    """
    try:
        if "1.5" in task_str:
            region, gt_id, resolution, horizon =  task_str.split('_')
            gt_id = f"{gt_id}_{resolution}"
        else:
            region, gt_id, horizon = task_str.split('_')
            
        if region not in ["contest", "us"]:
            raise ValueError("Bad region.")

        if gt_id not in ["tmp2m", "precip", "tmp2m_1.5x1.5", "precip_1.5x1.5"]:
            raise ValueError("Bad gt_id.")

        if horizon not in ["12w", "34w", "56w"]:
            raise ValueError("Bad horizon.")

    except Exception as e:
        printf("Could not get task parameters from task string.")
        return None

    return region, gt_id, horizon


def num_available_cpus():
    """Returns the number of CPUs available considering the sched_setaffinity
    Linux system call, which limits which CPUs a process and its children
    can run on.
    """
    return len(os.sched_getaffinity(0))


def hash_strings(strings, sort_first=True):
    """Returns a string hash value for a given list of strings.
    Always returns the same value for the same inputs.

    Args:
      strings: list of strings to hash
      sort_first: sort string list before hashing? if True, returns the same
        hash for the same collection of strings irrespective of their ordering
    """
    if sort_first:
        strings = sorted(strings)
    # Setting environment variable PYTHONHASHSEED to 0 disables hash randomness
    # Must be done prior to program execution, so we call out to a new Python
    # process
    return subprocess.check_output(
        "export PYTHONHASHSEED=0 && python -c \"print(str(abs(hash('{}'))))\"".format(
            ",".join(strings)), shell=True, universal_newlines=True).strip()


def string_to_dt(string):
    """Transforms string to datetime."""
    return datetime.datetime.strptime(string, DATETIME_FORMAT)


def dt_to_string(dt):
    """Transforms datetime to string."""
    return datetime.datetime.strftime(dt, DATETIME_FORMAT)


def get_dt_range(base_date, days_ahead_start=0, days_ahead_end=0):
    """Lists the dates between (base_date + days_ahead_start), included, and
    (base_date + days_ahead_end), not included.

    Parameters
    ----------
    base_date : datetime
        Reference date for time window.
    days_ahead_start : int
        Time window starts days_ahead_start days from base_date (included).
    days_ahead_end : int
        Time window ends days_ahead_start days from base_date (included).

    Returns
    -------
    List
        Dates in (base_date + days_ahead_start) and (base + days_ahead_end)

    """
    date_start = base_date + datetime.timedelta(days=days_ahead_start)
    days_in_window = days_ahead_end - days_ahead_start

    date_list = [date_start + datetime.timedelta(days=day)
                 for day in range(days_in_window)]
    return(date_list)


def get_current_year():
    """Gets year at the time when the script is run."""
    now = datetime.datetime.now()
    return now.year

