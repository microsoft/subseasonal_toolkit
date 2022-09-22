# Util functions for tuner model

import importlib
import os
import numpy as np
import pandas as pd
import subprocess
from glob import glob
import random
from itertools import product
from datetime import datetime, timedelta
from subseasonal_toolkit.utils.eval_util import get_target_dates, mean_rmse_to_score
from subseasonal_toolkit.utils.general_util import printf, make_directories, symlink
from subseasonal_toolkit.utils.experiments_util import get_first_year, get_start_delta
from subseasonal_toolkit.utils.models_util import (get_submodel_name, start_logger, log_params, get_forecast_filename,
                                                   save_forecasts)


MODEL_NAME = "tuner"
ONE_WEEK = timedelta(days=7)
ONE_DAY = timedelta(days=1)
ONE_YEAR = timedelta(days=365)
#Define functions




###############################################################################
    #Data related util functions
############################################################################### 

def get_model_selected_submodel_name(model_name="climpp",
                                     gt_id="contest_tmp2m",
                                     target_horizon="34w"):
    """Returns the string selected submodel name for a given model, gt_id and target_horizon.

    Args:
      model_name: string model name
      gt_id: contest_tmp2m or contest_precip
      target_horizon: 34w or 56w
    """
    get_selected_submodel_name = getattr(importlib.import_module("src.models."+model_name+".attributes"), "get_selected_submodel_name")
    return get_selected_submodel_name(gt_id, target_horizon)



def are_forecasts_missing(gt_id="contest_tmp2m", target_horizon="34w", model_name="tuned_climpp", submodel_name="climpp_years1_margin1", target_dates="std_train"):
    log_path = os.path.join("models", model_name, "submodel_forecasts", submodel_name, f"{gt_id}_{target_horizon}", "logs", f"{gt_id}_{target_horizon}-{target_dates}.log")
    #print(log_path)
    last_date_str = datetime.strftime(get_target_dates(target_dates)[-1],'%Y%m%d')
    #print(last_date_str)
    log_complete = False
    if os.path.exists(log_path):
        log_data = open(log_path, "r")
        for line in log_data:
            if "target date" in line and last_date_str in line:
                #print(line)
                log_complete = True
                break
    are_missing = False if (os.path.exists(log_path) and log_complete) else True
    return are_missing


    
def get_tuned_model_selected_submodel_name(output_model_name="tuned_climpp",
                                     gt_id="contest_tmp2m",
                                     target_horizon="34w"):
    """Returns the string selected submodel name for a given model, gt_id and target_horizon.

    Args:
      model_name: string model name
      gt_id: contest_tmp2m or contest_precip
      target_horizon: 34w or 56w
    """
    get_selected_submodel_name = getattr(importlib.import_module("src.models."+output_model_name+".attributes"), "get_selected_submodel_name")
    return get_selected_submodel_name(gt_id, target_horizon)    
    
def get_tuner_submodel_name(output_model_name="tuned_climpp", num_years=10, margin_in_days=45):
    """Returns submodel name for a given setting of model parameters
    """
    return f"{output_model_name}_on_years{num_years}_margin{margin_in_days}"    
    
    
def get_target_dates_all(gt_id="contest_tmp2m"):
    start_year = get_first_year(gt_id)
    start_date = datetime.strptime(f"{start_year}0101", "%Y%m%d")
    return pd.date_range(start = start_date, end = datetime.today()).to_pydatetime().tolist()


def load_metric_df(gt_id="contest_tmp2m", target_horizon="34w", model_name="climpp", metric="rmse",
                   metric_file_regex="*paper_eval*", first_year=2007):
    #STEP 1: get predict_rmse selected submodel
    # Store submodel or model performances in dataframe\n,
    task = f"{gt_id}_{target_horizon}"
    # Get metrics dataframe 
    metric_df = pd.DataFrame(index=get_target_dates_all(gt_id=gt_id))
    metric_df = metric_df[metric_df.index.year>=first_year]
    index_cols = 'start_date'
    # Identify the submodels with stored metric for this task
    eval_dir = os.path.join("eval", "metrics", model_name, "submodel_forecasts")
 
    #models_dir = glob(f"models/{model_name}/submodel_forecasts/*/{task}")
    models_dir = glob(f"{eval_dir}/*/{task}")
    submodel_names = []
    for d in models_dir:
        submodel_names.append(d.split(os.path.sep)[4])
    for submodel_name in submodel_names:
        filenames = glob(os.path.join(eval_dir, submodel_name, task, f"{metric}-{task}-{metric_file_regex}.h5"))
        if filenames:
            for f in filenames:
                df = pd.read_hdf(f) if filenames.index(f)==0 else df.append(pd.read_hdf(f))
            df = df.sort_values(by="start_date").set_index(index_cols).rename(columns={metric:submodel_name})
            df = df[~df.index.duplicated(keep='first')]
            metric_df = metric_df.join(df)
    metric_df = metric_df.dropna(axis=0, how="all").drop_duplicates()
    return metric_df


def get_tuning_dates(gt_id, horizon, target_date_obj, num_years, margin_in_days, X):
    """Returns indicator of whether each index element of X should be used for tuning
    
    Args:
        target_date_obj - target date datetime object
        X - dataframe with index equal to all relevant target dates and columns
            ["delta", "dividend", "remainder"]; warning: X will be modified!
    """
    start_delta = timedelta(days=get_start_delta(horizon, gt_id))
    last_train_date = target_date_obj - start_delta
    days_per_year = 365.242199

    X['delta'] = (target_date_obj - X.index).days
    X['remainder'] = np.floor(X.delta % days_per_year) 

    # Restrict data based on training date, dividend, and remainder
    indic = (X.index <= last_train_date)
    if margin_in_days is not None:
        indic &= ((X.remainder <= margin_in_days) | (X.remainder >= 365-margin_in_days))
    if num_years != "all":
        X['dividend'] = np.floor(X.delta / days_per_year)
        indic &= (X.dividend < num_years)

    return indic


def get_tuning_dates_selected_submodel(metric_df, tuning_dates):
    
    return metric_df[metric_df.index.isin(tuning_dates)].mean().idxmin() 




