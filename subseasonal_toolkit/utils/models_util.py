# Utility functions supporting models
import os
import sys
import pandas as pd
from importlib import import_module
from datetime import datetime, timedelta
from .general_util import printf, make_parent_directories, symlink
from .experiments_util import pandas2hdf
import inspect

def get_selected_submodel_name(model="spatiotemporal_mean",
                               gt_id="contest_tmp2m", horizon="34w", return_params=False,
                                target_dates=None):
    """Returns the name of the submodel selected by a given model;
    returns None if model has not exposed get_selected_submodel_name
    in subseasonal_toolkit.models.{model}.attributes. If return_params is True, returns
    a dictionary object containing selected submodel parameters if the 
    get_selected_submodel_name supports.

    Args:
       model: string model name
       gt_id: "contest_tmp2m" or "contest_precip"
       horizon: "12w", "34w", or "56w"
       return_params: True or False
    """
    try:
        attr = import_module(f"subseasonal_toolkit.models.{model}.attributes")
   
        # Check if modules get_seleted_selected_submodel name supports return_params
        has_return = "return_params" in inspect.getargspec(attr.get_selected_submodel_name).args
        has_target = "target_dates" in inspect.getargspec(attr.get_selected_submodel_name).args

        if has_return and has_target:
            return attr.get_selected_submodel_name(gt_id, horizon, return_params, target_dates)   
        else:
            return attr.get_selected_submodel_name(gt_id, horizon)   
    except (ModuleNotFoundError, AttributeError, FileNotFoundError, KeyError) as e:
        return None

def get_submodel_name(model, *args, **kwargs):
    """Returns the name of the submodel associated with a given model
    and a given set of positional and keyword arguments;
    returns None if model has not exposed get_submodel_name
    in subseasonal_toolkit.models.{model}.attributes

    Args:
       model: string model name
    """
    try:
        attr = import_module(f"subseasonal_toolkit.models.{model}.attributes")
        return attr.get_submodel_name(*args, **kwargs)
    except (ModuleNotFoundError, AttributeError, FileNotFoundError, KeyError) as e:
        return None
    
def get_d2p_submodel_names(model, *args, **kwargs):
    """Returns list of submodel names to be used in forming a probabilistic
    forecast using the d2p model

    Args:
      model: string model name
      gt_id: ground truth identifier in {"contest_tmp2m", "contest_precip"}
      target_horizon: string in {"34w", "56w"}
    """
    try:
        attr = import_module(f"subseasonal_toolkit.models.{model}.attributes")
        return attr.get_d2p_submodel_names(*args, **kwargs)
    except (ModuleNotFoundError, AttributeError, FileNotFoundError, KeyError) as e:
        return None


def get_task_forecast_dir(model="spatiotemporal_mean",
                          submodel=None,
                          gt_id="contest_tmp2m",
                          horizon="34w",
                          target_dates=None):
    """Returns the directory in which forecasts from a given submodel or
    model and a given task are stored

    Args:
       model: string model name
       submodel: string submodel name or None; if None, returns forecast
         directory associated with selected submodel or None if no
         submodel selected
       gt_id: contest_tmp2m or contest_precip
       horizon: "12w", "34w", or "56w"
    """
    if submodel is None:
        submodel = get_selected_submodel_name(model=model,gt_id=gt_id,
                                              horizon=horizon, target_dates=target_dates)
        if submodel is None:
            return None
    return os.path.join("models", model, "submodel_forecasts", submodel,
                        f"{gt_id}_{horizon}")

def get_forecast_filename(model="spatiotemporal_mean",
                          submodel=None,
                          gt_id="contest_tmp2m",
                          horizon="34w",
                          target_date_str="20191029"):
    """Returns the filename for storing forecasts from a given submodel or
    model for a given task and target date

    Args:
       model: string model name
       submodel: string submodel name or None; if None, returns forecast
         directory associated with selected submodel or None if no
         submodel selected
       gt_id: contest_tmp2m or contest_precip
       horizon: "12w", "34w", or "56w"
       target_date_str: first date of target two week period; string in
         YYYYMMDD format
    """
    outdir = get_task_forecast_dir(model=model, submodel=submodel,
                                gt_id=gt_id, horizon=horizon)
    return os.path.join(outdir, f"{gt_id}_{horizon}-{target_date_str}.h5")

def save_forecasts(preds,
                   model="spatiotemporal_mean",
                   submodel="spatiotemporal_mean-1981_2010",
                   gt_id="contest_tmp2m",
                   horizon="34w",
                   target_date_str="20191029"):
    """Saves predictions produced by a given model and submodel
    for a given target date and task

    Args:
       preds: pandas DataFrame with columns ['lat','lon','start_date','pred']
         containing predictions for the given target date
       model: string model name
       submodel: string submodel name or None; if None, returns forecast
         directory associated with selected submodel or None if no
         submodel selected
       gt_id: contest_tmp2m or contest_precip
       horizon: "12w", "34w", or "56w"
       target_date_str: first date of target two week period; string in
         YYYYMMDD format
    """
    outfile = get_forecast_filename(model=model, submodel=submodel,
                                    gt_id=gt_id, horizon=horizon,
                                    target_date_str=target_date_str)
    pandas2hdf(preds, outfile)

class Logger(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()

def get_log_filename(model="spatiotemporal_mean",
                     submodel=None,
                     gt_id="contest_tmp2m",
                     horizon="34w",
                     target_dates="std_val"):
    """Returns the filename for storing log for a given submodel or
    model for a given task and target dates

    Args:
       model: string model name
       submodel: string submodel name or None; if None, returns forecast
         directory associated with selected submodel or None if no
         submodel selected
       gt_id: contest_tmp2m or contest_precip
       horizon: "12w", "34w", or "56w"
       target_dates: string representing set of target dates
    """
    outdir = get_task_forecast_dir(model=model, submodel=submodel,
                                   gt_id=gt_id, horizon=horizon)
    return os.path.join(outdir, "logs",
                        f"{gt_id}_{horizon}-{target_dates}.log")

def start_logger(model="spatiotemporal_mean",
                 submodel=None,
                 gt_id="contest_tmp2m",
                 horizon="34w",
                 target_dates="std_val"):
    """Initializes and returns Logger object pointing to log file for the given
    model, submodel, task, and target_dates

    Args:
       model: string model name
       submodel: string submodel name or None; if None, selected submodel of
         model is used
       gt_id: contest_tmp2m or contest_precip
       horizon: "12w", "34w", or "56w"
       target_dates: string representing set of target dates
    """
    log_file = get_log_filename(model=model,submodel=submodel,gt_id=gt_id,
                                horizon=horizon,target_dates=target_dates)
    make_parent_directories(log_file)
    # 'w' overwrites previous log
    return Logger(log_file, 'w')

def log_params(params_names, params_values):
    """Log arguments using names in params_names and values in params_values.
    Args
    ----------
    params_names : list
        List with names of parameters to be logged
    params_values : list
        List with values of parameters to be logged
    """

    printf("Parameter values:")

    assert len(params_names)==len(params_values)

    for name, value in zip(params_names, params_values):
        printf(f"  {name}: {value}")

def get_task_contest_forecast_dir(model="spatiotemporal_mean",
                                  gt_id="contest_tmp2m",
                                  horizon="34w",
                                  target_dates=None):
    """Returns the directory for storing contest forecasts from a given
    model for a given task and target date

    Args:
       model: string model name
       gt_id: contest_tmp2m or contest_precip
       horizon: "12w", "34w", or "56w"
    """
    task = gt_id+"_"+horizon
    return os.path.join("models",model,"contest_forecasts",task)

def get_contest_forecast_filename(model="spatiotemporal_mean",
                                  gt_id="contest_tmp2m",
                                  horizon="34w",
                                  target_date_str="20191029"):
    """Returns the filename for storing contest forecasts from a given
    model for a given task and target date

    Args:
       model: string model name
       gt_id: contest_tmp2m or contest_precip
       horizon: "12w", "34w", or "56w"
       target_date_str: first date of target two week period; string in
         YYYYMMDD format
    """
    task = gt_id+"_"+horizon
    outdir = get_task_contest_forecast_dir(
        model=model,gt_id=gt_id,horizon=horizon)
    return os.path.join(outdir,task+"-"+target_date_str+".h5")

def load_contest_forecasts(model="spatiotemporal_mean",
                           gt_id="contest_tmp2m",
                           horizon="34w",
                           target_date_str="20191029"):
    """Returns the saved contest predictions produced by a given model
    for a given target date and task.

    Args:
       model: string model name
       gt_id: contest_tmp2m or contest_precip
       horizon: "12w", "34w", or "56w"
       target_date_str: first date of target two week period; string in
         YYYYMMDD format
    """
    in_file = get_contest_forecast_filename(
        model=model,gt_id=gt_id,horizon=horizon,
        target_date_str=target_date_str)
    printf("Reading "+in_file)
    return pd.read_hdf(in_file)

def get_contest_log_filename(model="spatiotemporal_mean",
                             gt_id="contest_tmp2m",
                             horizon="34w",
                             target_date_str="20191029"):
    """Returns the filename for storing log for a given
    model for a given task and target dates

    Args:
       model: string model name
       gt_id: contest_tmp2m or contest_precip
       horizon: "12w", "34w", or "56w"
       target_date_str: first date of target two week period; string in
         YYYYMMDD format
    """
    outdir = get_task_contest_forecast_dir(
        model=model,gt_id=gt_id,horizon=horizon)
    return os.path.join(outdir, "logs",
                        f"{gt_id}_{horizon}-{target_date_str}.log")

def link_contest_forecasts(model="spatiotemporal_mean",
                           gt_id="contest_tmp2m",
                           horizon="34w",
                           target_date_str="20191029"):
    """For the given model, task, and target date,
    1. Adds link inside of contest forecast folder to this model's forecast file
    2. Adds link inside of contest forecast log folder to this model's log file

    Args:
       model: string model name
       gt_id: contest_tmp2m or contest_precip
       horizon: "12w", "34w", or "56w"
       target_date_str: first date of target two week period; string in
         YYYYMMDD format
    """
    # Link to appropriate forecast in contest forecast folder
    src_file = get_forecast_filename(
        model=model,submodel=None,gt_id=gt_id,
        horizon=horizon,target_date_str=target_date_str)
    dest_file = get_contest_forecast_filename(
        model=model,gt_id=gt_id,
        horizon=horizon,target_date_str=target_date_str)
    make_parent_directories(dest_file)
    printf(f"Soft-linking\n-dest: {dest_file}\n-src: {src_file}")
    symlink(src_file, dest_file, use_abs_path=True)

    # Link to appropriate forecast log in contest forecast log folder
    src_file = get_log_filename(
        model=model,submodel=None,gt_id=gt_id,
        horizon=horizon,target_dates=target_date_str)
    dest_file = get_contest_log_filename(
        model=model,gt_id=gt_id,
        horizon=horizon,target_date_str=target_date_str)
    make_parent_directories(dest_file)
    printf(f"Soft-linking\n-dest: {dest_file}\n-src: {src_file}")
    symlink(src_file, dest_file, use_abs_path=True)
