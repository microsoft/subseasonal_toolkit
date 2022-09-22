#!/usr/bin/env python
# coding: utf-8

# # N-BEATS
# 
# ### Uses nbeats model as prediction of future.

# In[1]:


import os, sys
from tqdm import tqdm
from subseasonal_toolkit.utils.notebook_util import isnotebook
if isnotebook():
    # Autoreload packages that are modified
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
else:
    from argparse import ArgumentParser
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist, euclidean
from datetime import datetime, timedelta
from ttictoc import tic, toc
from subseasonal_data.utils import get_measurement_variable
from subseasonal_toolkit.utils.general_util import printf
from subseasonal_toolkit.utils.experiments_util import get_first_year, get_start_delta
from subseasonal_toolkit.utils.models_util import (get_submodel_name, start_logger, log_params, get_forecast_filename,
                                                   save_forecasts)
from subseasonal_toolkit.utils.eval_util import get_target_dates, mean_rmse_to_score, save_metric
from sklearn.linear_model import *

from subseasonal_data import data_loaders


# In[ ]:


#
# Specify model parameters
#
if not isnotebook():
    # If notebook run as a script, parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument("pos_vars",nargs="*")  # gt_id and horizon                                                                                  
    parser.add_argument('--target_dates', '-t', default="std_test")
    args, opt = parser.parse_known_args()
    
    # Assign variables                                                                                                                                     
    gt_id = args.pos_vars[0] # "contest_precip" or "contest_tmp2m"                                                                            
    horizon = args.pos_vars[1] # "12w", "34w", or "56w"    
    target_dates = args.target_dates
else:
    # Otherwise, specify arguments interactively 
    gt_id = "contest_tmp2m"
    horizon = "34w"
    target_dates = "std_contest"

#
# Process model parameters
#
# One can subtract this number from a target date to find the last viable training date.
start_delta =  timedelta(days=get_start_delta(horizon, gt_id))

# Record model and submodel name
model_name = "nbeats"
submodel_name = get_submodel_name(model_name)

FIRST_SAVE_YEAR = 2007 # Don't save forecasts from years prior to FIRST_SAVE_YEAR

if not isnotebook():
    # Save output to log file
    logger = start_logger(model=model_name,submodel=submodel_name,gt_id=gt_id,
                          horizon=horizon,target_dates=target_dates)
    # Store parameter values in log                                                                                                                        
    params_names = ['gt_id', 'horizon', 'target_dates']
    params_values = [eval(param) for param in params_names]
    log_params(params_names, params_values)


# In[ ]:


printf('Loading target variable and dropping extraneous columns')
tic()
var = get_measurement_variable(gt_id)
gt = data_loaders.get_ground_truth(gt_id).loc[:,["start_date","lat","lon",var]]
toc()


# In[ ]:


printf('Pivoting dataframe to have one column per lat-lon pair and one row per start_date')
tic()
gt = gt.set_index(['lat','lon','start_date']).squeeze().unstack(['lat','lon'])
toc()


# In[ ]:


#
# Make predictions for each target date
#
from fbprophet import Prophet
from pandas.tseries.offsets import DateOffset

def get_first_fourth_month(date):
    targets = {(1, 31), (3,31), (5, 31), (7, 31), (9, 30), (11,30)}
    while (date.month, date.day) not in targets:
        date = date - DateOffset(days=1)
    return date
from dateutil.relativedelta import relativedelta

def get_predictions(date):
    # take the first (12/31, 8/31, 4/30) right before the date. 
    true_date = get_first_fourth_month(date)
    if horizon == "34w":
        prediction_lag = 28
    if horizon == "56w":
        prediction_lag = 42
    shifts = np.flip(np.arange(0, 13)) * 4
    cmd_prefix = "python main.py --features "
    cmd_features = ""
    for shift in shifts:
        cmd_features = cmd_features + f"{gt_id}_shift{shift} "
    a = len(cmd_features)
    cmd_features += f'\\'
    b = len(cmd_features)
    epochs = 20
    cnt = 0
    target_date = (true_date - timedelta(days=prediction_lag) - relativedelta(years=1)).strftime("%Y-%m-%d")
    val_begin_date = (true_date - timedelta(days=prediction_lag) - relativedelta(years=1) + timedelta(days=1)).strftime("%Y-%m-%d")
    val_end_date = (true_date - timedelta(days=prediction_lag)).strftime("%Y-%m-%d")
    test_begin_date = true_date.strftime("%Y-%m-%d")
    test_end_date = (true_date + relativedelta(months=2) -timedelta(days=1)).strftime("%Y-%m-%d")
    log_params(["target_date"], [target_date])
    converted_date = date.strftime('%Y_%m_%d')
    cmd_suffix = f"--task '{gt_id}_{horizon}' --train --evaluate --path-to-feather-file 'data/gt-{gt_id}-14d.h5' --begin-date 1979-01-01 \    --target-date {target_date} \    --val-begin-date {val_begin_date} \    --val-end-date {val_end_date} \    --test-begin-date {test_begin_date} --test-end-date {test_end_date}  \    --batch-size 512 --max_epochs {epochs} --grid-count -1 --gpus 1 --gpu-idx {cnt} --full \    --default_root_dir {gt_id}_{horizon}_exp_{converted_date} " 
    cmd = cmd_prefix + cmd_features + cmd_suffix
    log_params(["cmd"], [cmd])
    print(cmd)
    os.system(cmd) # comment to not run the actual program.

    # open the file where this is outputted. 
    date_string = date.strftime("%Y%m%d")
    filename = f"nbeats-predictions/{gt_id}_{horizon}-{date_string}.h5"
    # return the answer. 
    return pd.read_hdf(filename).values

tic()
target_date_objs = pd.Series(get_target_dates(date_str=target_dates,horizon=horizon))
rmses = pd.Series(index=target_date_objs, dtype=np.float64)
preds = pd.DataFrame(index = target_date_objs, columns = gt.columns, 
                     dtype=np.float64)
preds.index.name = "start_date"
# Sort target_date_objs by day of week
target_date_objs = target_date_objs[target_date_objs.dt.weekday.argsort(kind='stable')]
toc()
for target_date_obj in target_date_objs:
    tic()
    target_date_str = datetime.strftime(target_date_obj, '%Y%m%d')
    # Find the last observable training date for this target
    last_train_date = target_date_obj - start_delta
    if not last_train_date in gt.index:
        printf(f'-Warning: no persistence prediction for {target_date_str}; skipping')
        continue
    printf(f'Forming persistence prediction for {target_date_obj}')

    # key logic here:
    
    preds.loc[target_date_obj,:] = get_predictions(target_date_obj)
    break
    # Save prediction to file in standard format
    if target_date_obj.year >= FIRST_SAVE_YEAR:
        save_forecasts(
            preds.loc[[target_date_obj],:].unstack().rename("pred").reset_index(),
            model=model_name, submodel=submodel_name, 
            gt_id=gt_id, horizon=horizon, 
            target_date_str=target_date_str)
    # Evaluate and store error if we have ground truth data
    if target_date_obj in gt.index:
        rmse = np.sqrt(np.square(preds.loc[target_date_obj,:] - gt.loc[target_date_obj,:]).mean())
        rmses.loc[target_date_obj] = rmse
        print("-rmse: {}, score: {}".format(rmse, mean_rmse_to_score(rmse)))
        mean_rmse = rmses.mean()
        print("-mean rmse: {}, running score: {}".format(mean_rmse, mean_rmse_to_score(mean_rmse)))
    toc()

printf("Save rmses in standard format")
rmses = rmses.sort_index().reset_index()
rmses.columns = ['start_date','rmse']
save_metric(rmses, model=model_name, submodel=submodel_name, gt_id=gt_id, horizon=horizon, target_dates=target_dates, metric="rmse")

