"""Utility functions supporting salient experiments


Attributes:
    IN_DIR (str): input directory for salient related files.
    INDIR (str): input directory for salient related files.  
    window_size (int): size of the sliding window over the data. If set to 10, 
        the NN's input feature vector consists of a concatenation of the prior 10 weeks of data.
    WEEKS (int): = size of the sliding window over the data. If set to 10, 
        the NN's input feature vector consists of a concatenation of the prior 10 weeks of data.
    ONE_WEEK (datetime.timedelta):  A duration of seven days between two datetime dates.
    ONE_DAY (datetime.timedelta): A duration of one day between two datetime dates.
    
"""

from __future__ import print_function
import os
import sys
import errno
import numpy as np
import numpy.ma as ma
from netCDF4 import num2date
from scipy.interpolate import griddata
from datetime import datetime, timedelta


ONE_WEEK = timedelta(days=7)
ONE_DAY = timedelta(days=1)
IN_DIR = os.path.join("models", "salient")
INDIR = IN_DIR
window_size = 10
WEEKS = 10

# Set frequently used directories
dir_predict_data = os.path.join(IN_DIR, "predict-data")
dir_raw_processed = os.path.join(IN_DIR, "raw-processed")
dir_submodel_forecasts  = os.path.join(IN_DIR, "submodel_forecasts")
dir_train_data = os.path.join(IN_DIR, "train-data")
dir_train_results = os.path.join(IN_DIR, "train-results")

# Fix python2
try:
    input = raw_input
except NameError:
    pass

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via input() and return their answer.

    Args:
        question (str): is a string that is presented to the user.
        default (str): is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    Returns:
        True for "yes" or False for "no".

    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stderr.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stderr.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def get_target_date(ask_for=None, input_str=None):
    """Get first day of target 2-week period.

    Args:
        ask_for (str): is a string that is presented to the user.
        input_str (str): target date string formatted as "%Y-%m-%d" or "%Y%m%d".

    Returns:
        tuple containing target date datetime object and target date string.

    """
    
    target_date = None

    if input_str:
        try:
            target_date = datetime.strptime(input_str, "%Y-%m-%d").date()
        except ValueError:
            target_date = datetime.strptime(input_str, "%Y%m%d").date()

    if not target_date:
        if not ask_for:
            raise Exception('Unknown target date')
        # default to next Tuesday (weekday #1)
        today = datetime.now().date()
        days_ahead = (1 - today.weekday() + 7) % 7
        target_date = today + timedelta(days=days_ahead)
        if not query_yes_no(ask_for + ' for ' + str(target_date) + '?'):
            exit(1)

    return (target_date, target_date.strftime('%Y%m%d'))


def mkdir_p(path):
    """Creates a directory from path, if not already existing.

    Args:
        path (str): The path of the directory to be created.

    """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_date_range(timevar):
    """Obtain start and end date of a time variable in a netcdf file.

    Args:
        timevar (arr???): time variable in a netcdf file.

    Returns:
        tuple containing start and end dates of the time variable.
        

    """
    start = num2date(timevar[0], timevar.units)
    end = num2date(timevar[-1], timevar.units)
    return (start, end)

def year_fraction(date):
    """Obtain the fraction of the year that a given date represents.

    Args:
        date (datetime): a datetime object.

    Returns:
        float representing the fraction of the year.

    """
    year = date.year
    this_year_start = datetime(year=year, month=1, day=1)
    next_year_start = datetime(year=year+1, month=1, day=1)
    days_elapsed = date.timetuple().tm_yday - 0.5
    days_total = (next_year_start - this_year_start).days
    return days_elapsed/days_total


def val_i_error(y_true, y_pred, i):
    """returns the validation loss for a single training example of a keras NN model.

    Args:
        i: index of training example of a keras NN model.

    Returns:
        validation loss value for the ith training example.

    """
    start = i*514
    end = (i+1)*514
    y_true = y_true[...,start:end]
    y_pred = y_pred[...,start:end]
    err = np.abs(y_true - y_pred)
    mean_err = np.mean(err)
    return mean_err

################################################################################
# download_recent_data.py                                                      #
################################################################################
def date2datetime(date):
    """Convert a date to a datetime object.

    Args:
        date: date to be converted (double check type and use).

    Returns:
        date converted to object of type datetime.

    """
    return datetime.combine(date, datetime.min.time())


################################################################################
# predict_data_gen.py                                                          #
################################################################################

def ma2d_interp(array):
    """Interpolate a 2-dimensional masked array.

    Args:
        array (float): 2-dimensional array to be linearly interpolated.

    Returns:
        an interpolated masked array.

    """
    valid = np.where(array.mask == False)
    w = array.shape[0]
    h = array.shape[1]
    grd = np.array(np.mgrid[0:w, 0:h])
    grd2 = np.vstack((grd[0][valid], grd[1][valid]))
    pts = grd2.T.flatten().reshape(int(grd2.size/2), 2)
    return griddata(pts, array[valid].T.flatten(), (grd[0], grd[1]), method='linear')

def ma_interp(array):
    """Interpolate a 3-dimensional masked array.

    Args:
        array (float): 3-dimensional array to be linearly interpolated.

    Returns:
        an interpolated 3-dimensional masked array.

    """
    if len(array.shape) == 2:
        return ma2d_interp(array)
    if len(array.shape) == 3:
        # assume first dimention shouldn't be interpolated
        output = np.empty_like(array, subok=False)
        for i in range(array.shape[0]):
            output[i] = ma2d_interp(array[i])
    return output

def array2d_reduce(array, zoom):
    """Reduce a 2-dimensional masked array.

    Args:
        array (float): 2-dimensional array to be reduced.
        zoom (float): reducing factor.

    Returns:
        a 2-dimensional masked array.

    """
    output = ma.masked_all([int(array.shape[0] / zoom), int(array.shape[1] / zoom)])
    for i in range(int(array.shape[0] / zoom)):
        for j in range(int(array.shape[1] / zoom)):
            x = i * zoom
            y = j * zoom
            output[i, j] = array[x:x+zoom,y:y+zoom].mean()
    return output

def array_reduce(array, zoom):
    """Reduce a 3-dimensional masked array.

    Args:
        array (float): 3-dimensional array to be reduced.
        zoom: reducing factor.

    Returns:
        a reduced 3-dimensional masked array.

    """
    if len(array.shape) == 2:
        return array2d_reduce(array, zoom)
    if len(array.shape) == 3:
        # assume first dimention shouldn't be zoomed
        output = ma.masked_all([array.shape[0], int(array.shape[1] / zoom), int(array.shape[2] / zoom)])
        for i in range(array.shape[0]):
            output[i] = array2d_reduce(array[i], zoom)
        return output



