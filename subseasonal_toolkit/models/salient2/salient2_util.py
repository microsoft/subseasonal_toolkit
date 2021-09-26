"""Utility functions supporting salient2 experiments


Attributes:
    IN_DIR (str): input directory for salient2 related files.
    INDIR (str): input directory for salient2 related files.  
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
import subprocess
import numpy as np
import pandas as pd
import numpy.ma as ma
from netCDF4 import num2date
from scipy.interpolate import griddata
from datetime import datetime, timedelta
from pkg_resources import resource_filename
from subseasonal_data.utils import load_measurement
from subseasonal_toolkit.utils.general_util import printf


ONE_WEEK = timedelta(days=7)
ONE_DAY = timedelta(days=1)
IN_DIR = "models/salient2"
INDIR = "models/salient2"
window_size = 10
WEEKS = 10


# Set frequently used directories
dir_predict_data = os.path.join(IN_DIR, "predict-data")
dir_raw_processed = os.path.join(IN_DIR, "raw-processed")
dir_submodel_forecasts  = os.path.join(IN_DIR, "submodel_forecasts")
dir_train_data = os.path.join(IN_DIR, "train-data")
dir_train_results = os.path.join(IN_DIR, "train-results")

# Set training end dates for years 2006-2019 
training_end_dates = {'2006': '20060726',
                      '2007': '20070725',
                      '2008': '20080730',
                      '2009': '20090729',
                      '2010': '20100727',
                      '2011': '20110726',
                      '2012': '20120731',
                      '2013': '20130730',
                      '2014': '20140729',
                      '2015': '20150728',
                      '2016': '20160727',
                      '2017': '20170201',
                      '2018': '20180725',
                      '2019': '20190731'}

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
        # default to next Wednesday (weekday #2)
        today = datetime.now().date()
        days_ahead = (2 - today.weekday() + 7) % 7
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


# generate data utils
def create_input_time(date):
    # Create time vector as year fraction of the date vector 
    printf(f"Input times start on Wednesdays from {date[0][0]} to {date[-1][0]}")
    data = np.zeros((len(date), 1))
    for i in range(len(date)):
        dt = datetime(date[i][0].astype(object).year, date[i][0].astype(object).month, date[i][0].astype(object).day)
        data[i,0] = year_fraction(dt)
    return data


def create_input_sst(date):
    printf(f"Input sst data starts on weeks centered around Wednesdays from {date[0][0]} to {date[-1][0]}")

    # Create SST vectors from salient fri vectors up until 20170201 and from noaa data afterwards        
    data = pd.read_pickle(resource_filename("subseasonal_toolkit", os.path.join("models", "salient2", "data", "sfri_sst.pickle")))
    
    salient_fri_end_date = datetime.strptime("20170201", "%Y%m%d")
    salient_frii_end_date = datetime.strptime("20201231", "%Y%m%d")
    predict_data_dir = os.path.join("models", "salient2", "predict-data", "d2wk_cop_sst")
    for d in date:
        #printf(d)
        dt = datetime(d[0].astype(object).year, d[0].astype(object).month, d[0].astype(object).day)
        dt_str = datetime.strftime(dt, "%Y%m%d")
        if dt <= salient_fri_end_date:
            continue
        elif dt <= salient_frii_end_date:
            if os.path.isfile(os.path.join(predict_data_dir, dt_str, "sst.pickle")) is False:
                cmd_script = resource_filename("subseasonal_toolkit", os.path.join("models", "salient2", "predict_data_gen.py"))
                cmd = f"python {cmd_script} -d {dt_str}" 
                #print(cmd)
                subprocess.call(cmd, shell=True)
            dt_sst = pd.read_pickle(os.path.join(predict_data_dir, dt_str, "sst.pickle"))
            data = np.append(data, dt_sst[-1,:].reshape((1, data.shape[1])), axis=0)   
    return data

def create_input_mei(date):    
    # Create mei vector from NOAA data
    data = pd.read_hdf(os.path.join("data", "dataframes", "gt-mei.h5"))
    #offset by 5 weeks due to mei data lag being 30 days
    date_mei = [datetime(d[0].astype(object).year, d[0].astype(object).month, d[0].astype(object).day) - 5*ONE_WEEK for d in date]
    df_mei = pd.DataFrame(columns = ["start_date", "mei"])
    df_mei["start_date"] = date_mei
    df_mei.set_index('start_date', inplace=True)
    printf(f"Input mei data is offset by 5 weeks and starts on weeks centered around Wednesdays from {df_mei.index[0].date()} to {df_mei.index[-1].date()}")
    for d in df_mei.index:
        #salient fri's original data and dates is weekly centered around wednsdays
        data_wk = data.loc[data["start_date"]>=d-3*ONE_DAY]
        data_wk = data_wk.loc[data_wk["start_date"]<=d+3*ONE_DAY]
        df_mei.loc[d] = data_wk["mei"].mean()
    return df_mei.values
    
def create_input_mjo(date):        
    # Create mjo vector from bom.gov.au data    
    data = pd.read_hdf(os.path.join("data", "dataframes", "gt-mjo-1d.h5"))
    df_mjo = pd.DataFrame(columns = ["start_date", "phase", "amplitude"])
    df_mjo["start_date"] = [datetime(d[0].astype(object).year, d[0].astype(object).month, d[0].astype(object).day) for d in date]
    df_mjo.set_index('start_date', inplace=True)
    printf(f"Input mjo data starts on weeks centered around Wednesdays from {df_mjo.index[0].date()} to {df_mjo.index[-1].date()}")    
    for d in df_mjo.index:
        #salient fri's original data and dates is weekly centered around wednsdays
        data_wk = data.loc[data["start_date"]>=d-3*ONE_DAY]
        data_wk = data_wk.loc[data_wk["start_date"]<=d+3*ONE_DAY]
        df_mjo.loc[d] = data_wk[["phase", "amplitude"]].mean()
    return df_mjo.values     
 



def create_output_contest(date, gt_var):
    # Create temp and precip vectors from salient2 vectors up until 20170201 and from noaa data afterwards        
    #for gt_var in ['contest_latlons', 'contest_tmp2m', 'contest_precip']:
    gt_var_or = gt_var
    gt_var = "contest_tmp2m" if "latlon" in gt_var_or else gt_var  
    
    #Get output start and end dates
    #Output consists of weeks starting on Tuesdays for the contest data
    input_start_date = datetime(date[0][0].astype(object).year, date[0][0].astype(object).month, date[0][0].astype(object).day)
    input_end_date  = datetime(date[-1][0].astype(object).year, date[-1][0].astype(object).month, date[-1][0].astype(object).day)
    output_start_date = input_start_date - ONE_DAY 
    output_end_date = input_end_date - ONE_DAY 
    
    #load gt data
    # Infile and outfile
    in_file = os.path.join("data", "dataframes", f"gt-{gt_var}-7d.h5")
    # Load data, apply mask for tmp2m and precip only
    gt = load_measurement(in_file, None)
    # Transform to wide format
    print("Transforming to wide format")
    gt_wide = gt.set_index(['lat','lon','start_date']).unstack(['lat','lon'])
    data = pd.DataFrame(gt_wide.to_records())
    columns = [c for c in data.columns if "std" not in c and "sqd" not in c]
    data = data[columns]
    if "latlons" in gt_var_or:
        data_columns = [c for c in data.columns if c != 'start_date' and "std" not in c and "sqd" not in c]
        contest_latlons = pd.DataFrame()
        contest_lats = [round(float(c.split(",")[1])) for c in sorted(data_columns)]
        contest_lons = [round(float(c.split(",")[2].replace(")",""))) for c in sorted(data_columns)]
        contest_latlons["lat"] = contest_lats
        contest_latlons["lon"] = contest_lons
        data_all = contest_latlons
    else:
        #select relevant training dates
        data_columns = [c for c in data.columns if c != 'start_date'] 
        #Output consists of weeks starting on Tuesdays for the contest data
        contest_output_dates = [d for d in data["start_date"] if d.weekday() == 1]
        data = data.loc[data["start_date"].isin(contest_output_dates)]
        data = data.loc[(data["start_date"]>=output_start_date) & (data["start_date"]<=output_end_date)]
        printf(f"Output contest data starts on weeks starting on Tuesdays from {data['start_date'].iloc[0].date()} to {data['start_date'].iloc[-1].date()}")    
        data = data.loc[:, data_columns].values
        data_all = data.T   
    return data_all


def create_output_east(date, gt_var):
    # Create U.S. temp and precip vectors from salient fri vectors up until 20170201 and from noaa data afterwards        
    #elif gt_var in ['east_tmp2m', 'east_precip', 'east_latlons']:
    gt_var_or = gt_var
    gt_var = "us_tmp2m" if "latlon" in gt_var_or else gt_var
    
    # Infile and outfile
    in_file = os.path.join("data", "dataframes", f"gt-{gt_var.replace('east', 'us')}-7d.h5")
    # Load data, apply mask for tmp2m and precip only
    gt = load_measurement(in_file, None)
    #restric to east latlons
    contest_latlons_filename = os.path.join("models", "salient2", "train-data", "contest_latlons.h5")
    contest_latlons = pd.read_hdf(contest_latlons_filename)
    contest_latlons = list(zip(contest_latlons['lat'], contest_latlons['lon']))
    
    gt['latlon'] = list(zip(gt['lat'].astype('int').values, gt['lon'].astype('int').values))
    gt = gt[~gt['latlon'].isin(contest_latlons)].drop('latlon', axis=1)  
    # Transform to wide format
    print("Transforming to wide format")
    gt_wide = gt.set_index(['lat','lon','start_date']).unstack(['lat','lon'])
    data = pd.DataFrame(gt_wide.to_records())
    
    #select relevant training dates
    data_columns = [c for c in data.columns if c != 'start_date' and "std" not in c and "sqd" not in c]
    
    if "latlons" in gt_var_or:
        latlons = pd.DataFrame()
        lats = [round(float(c.split(",")[1])) for c in sorted(data_columns)]
        lons = [round(float(c.split(",")[2].replace(")",""))) for c in sorted(data_columns)]
        latlons["lat"] = lats
        latlons["lon"] = lons
        data_all = latlons
    else:
        #Output consists of weeks starting on Wednesdays for the contest data 
        data = data.loc[data["start_date"].isin(date[:,0])]
        printf(f"Output east data starts on weeks starting on Wednesdays from {data['start_date'].iloc[0].date()} to {data['start_date'].iloc[-1].date()}")
        data_all = data.loc[:, data_columns].values.T
    return data_all
       

def create_output_us(date, gt_var):
    # Create U.S. temp and precip vectors from salient fri vectors up until 20170201 and from noaa data afterwards        
    #elif gt_var in ['us_tmp2m', 'us_precip', 'us_latlons']:
    gt_var_or = gt_var
    gt_var = "us_tmp2m" if "latlon" in gt_var_or else gt_var
    
    # Infile and outfile
    in_file = os.path.join("data", "dataframes", f"gt-{gt_var}-7d.h5")
    # Load data, apply mask for tmp2m and precip only
    gt = load_measurement(in_file, None)
    # Transform to wide format
    print("Transforming to wide format")
    gt_wide = gt.set_index(['lat','lon','start_date']).unstack(['lat','lon'])
    data = pd.DataFrame(gt_wide.to_records())
    
    #select relevant training dates
    data_columns = [c for c in data.columns if c != 'start_date' and "std" not in c and "sqd" not in c]
    
    if "latlons" in gt_var_or:
        us_latlons = pd.DataFrame()
        us_lats = [round(float(c.split(",")[1])) for c in sorted(data_columns)]
        us_lons = [round(float(c.split(",")[2].replace(")",""))) for c in sorted(data_columns)]
        us_latlons["lat"] = us_lats
        us_latlons["lon"] = us_lons
        data_all = us_latlons
    else:
        #Output consists of weeks starting on Wednesdays for the contest data
        data = data.loc[data["start_date"].isin(date[:,0])]
        printf(f"Output U.S. data starts on weeks starting on Wednesdays from {data['start_date'].iloc[0].date()} to {data['start_date'].iloc[-1].date()}")
        data_all = data.loc[:, data_columns].values.T
    return data_all



def create_gt_var(gt_var, date):
    # Create input data  
    if gt_var == "date":
        data = date
    elif gt_var == "time":
        data = create_input_time(date)
    elif gt_var == "sst":
        data = create_input_sst(date)
    elif gt_var == "mei":
        data = create_input_mei(date)
    elif gt_var == "mjo":
        data = create_input_mjo(date)    
    # Create output data
    elif gt_var in ['contest_latlons', 'contest_tmp2m', 'contest_precip']:
        data = create_output_contest(date, gt_var)
    elif gt_var in ['east_tmp2m', 'east_precip', 'east_latlons']:
        data = create_output_east(date, gt_var)
    elif gt_var in ['us_tmp2m', 'us_precip', 'us_latlons']:
        data = create_output_us(date, gt_var)    
    return data
