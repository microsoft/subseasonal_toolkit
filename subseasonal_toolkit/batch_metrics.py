# Evaluate error metrics for model or submodel for a specified set of test
# dates. Output is stored in eval/metrics/MODEL_NAME/SUBMODEL_NAME. If no
# submodel is provided, the selected submodel for a model is evaluated.
#
# Example usage:
#   python -m subseasonal_toolkit.batch_metrics contest_tmp2m 34w -mn climpp -t std_paper
#   python -m subseasonal_toolkit.batch_metrics contest_precip 56w -mn climpp -sn climpp-lossmse_years26_margin0 -t std_paper
#   python -m subseasonal_toolkit.batch_metrics contest_precip 56w -mn climpp -t std_paper -m lat_lon_rmse
#   python -m subseasonal_toolkit.batch_metrics contest_precip 56w -mn climpp -t std_paper -m rmse score skill lat_lon_rmse
#
# Positional args:
#   gt_id: contest_tmp2m, contest_precip, us_tmp2m, us_precip
#   horizon: 34w or 56w
#
# Named args:
#   --target_dates (-t): target dates for batch prediction (e.g.,
#     'std_val','std_paper','std_ens') (default: 'std_paper')
#   --metrics (-m): Space-separated list of error metrics to compute (valid choices are rmse, score, skill, lat_lon_rmse, lat_lon_skill, mse);
#     computes rmse, score, and skill by default
#   --model_name (-mn): name of model, e.g, climpp (default: None)
#   --submodel_name (-sn):  name of submodel, e.g., spatiotemporal_mean-1981_2010
#       (default: None)

import os
import subprocess
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from filelock import FileLock
from argparse import ArgumentParser
from subseasonal_data.utils import get_measurement_variable
from subseasonal_data.data_loaders import get_ground_truth, get_climatology
from subseasonal_toolkit.utils.general_util import printf, make_directories, tic, toc
from subseasonal_toolkit.utils.experiments_util import pandas2hdf
from subseasonal_toolkit.utils.eval_util import (get_target_dates, mean_rmse_to_score, 
                                                 get_task_metrics_dir)
from subseasonal_toolkit.utils.models_util import get_task_forecast_dir, get_selected_submodel_name

# Load command line arguments
parser = ArgumentParser()
parser.add_argument("pos_vars", nargs="*")  # gt_id and horizon
parser.add_argument('--target_dates', '-t', default='std_paper')
# For metrics, 1 or more values expected => creates a list
parser.add_argument("--metrics", '-m', nargs="+", type=str, default=['rmse', 'score', 'skill', 'lat_lon_rmse', 'lat_lon_skill'], help="Space-separated list of error metrics to compute (valid choices are rmse, score, skill, anom, error, lat_lon_rmse, lat_lon_skill, lat_lon_anom, lat_lon_pred, mse, wtd_mse)")
parser.add_argument('--model_name', '-mn', default=None)
parser.add_argument('--submodel_name', '-sn', default=None)
args = parser.parse_args()

# Assign variables
gt_id = args.pos_vars[0]  # e.g., "contest_precip" or "contest_tmp2m"
horizon = args.pos_vars[1]  # e.g., "12w", "34w", or "56w"
target_dates = args.target_dates
metrics = args.metrics
model_name = args.model_name
submodel_name = args.submodel_name

# Set input folder (with pred files) and output folder (for metrics)
if model_name == 'gt':
    submodel_name = model_name

# Set input folder (with pred files) and output folder (for metrics)
preds_folder = get_task_forecast_dir(
    model=model_name, submodel=submodel_name, gt_id=gt_id, horizon=horizon,
    target_dates=target_dates)
output_folder = get_task_metrics_dir(
    model=model_name, submodel=submodel_name, gt_id=gt_id, horizon=horizon,
    target_dates=target_dates)


# Get preds filenames
printf('Getting prediction file paths and target dates')
tic()
# Use set of test dates to determine which preds dfs to load
target_date_objs = get_target_dates(date_str=target_dates, horizon=horizon)
file_names = [f"{gt_id}_{horizon}-{datetime.strftime(target_date, '%Y%m%d')}.h5"
              for target_date in target_date_objs]

# Get list of sorted preds file paths and target dates
file_names = sorted(file_names)
file_paths = [f"{preds_folder}/{file_name}" for file_name in file_names]
# Extract date from file name as the penultimate list element 
# after splitting on periods and dashes
target_date_strs = [file_name.replace(
    '-', '.').split('.')[-2] for file_name in file_names]
target_date_objs = [datetime.strptime(
    date_str, '%Y%m%d') for date_str in target_date_strs]
toc()

# Load gt dataframe
printf('Loading ground truth')
tic()
var = get_measurement_variable(gt_id)
gt = get_ground_truth(gt_id).loc[:,['lat', 'lon', 'start_date', var]]
gt = gt.loc[gt.start_date.isin(target_date_objs),:].set_index(['start_date', 'lat', 'lon']).squeeze().sort_index()
toc()

# Create error dfs; populate start_date column with target_date_strs
metric_dfs = {}
for metric in metrics:
    if metric == 'lat_lon_rmse':
        # Keep track of number of dates contributing to error calculation
        # Initialize dataframe later
        num_dates = 0
        continue
    # Index by target dates
    metric_dfs[metric] = pd.Series(name=metric, index=target_date_objs, dtype=np.float64)
    metric_dfs[metric].index.name = 'start_date'
    if metric == 'wtd_mse':
        weights = None
    if 'skill' in metric or 'lat_lon' in metric or 'anom' in metric:
        # Load climatology
        printf('Loading climatology and replacing start date with month-day')
        tic()
        clim = get_climatology(gt_id)
        clim = clim.set_index(
            [clim.start_date.dt.month,clim.start_date.dt.day,'lat','lon']
        ).drop(columns='start_date').squeeze().sort_index()
        toc()
    if metric == 'lat_lon_skill':
        # Keep track of number of dates contributing to error calculation
        # Initialize dataframe later
        num_dates_lls = 0
    if metric == 'lat_lon_anom':
        # Keep track of number of dates contributing to error calculation
        # Initialize dataframe later
        num_dates_lla = 0
    if metric == 'lat_lon_pred':
        # Keep track of number of dates contributing to error calculation
        # Initialize dataframe later
        num_dates_llp = 0
    if metric == 'lat_lon_error':
        # Keep track of number of dates contributing to error calculation
        # Initialize dataframe later
        num_dates_lle = 0

# Fill the error dfs for given target date
def get_rmse(pred, gt):
    return np.sqrt(np.square(pred-gt).mean())

def get_mse(pred, gt):
    return np.square(pred-gt).mean()

def get_wtd_mse(pred, gt, weights):
    # Returns weighted average of squared errors over non-na values: 
    # sum_i weights_i (pred_i - gt_i)^2/sum_i weights_i
    vals = np.square(pred-gt)
    not_nas = vals.notna()
    return np.average(vals[not_nas], weights = weights[not_nas])

def get_skill(pred, gt, clim):
    return 1 - cosine(pred-clim, gt-clim)

def get_anom(pred, clim):
    return (pred-clim).mean()

def get_error(pred, gt):
    return (pred-gt).mean()

tic()
for file_path, target_date_obj in zip(file_paths, target_date_objs):
    printf(f'Getting metrics for {target_date_obj}')
    target_date_str = datetime.strftime(target_date_obj, '%Y%m%d')
    printf(target_date_str)
    tic()
    if target_date_obj not in gt.index:
        printf(f"Warning: {target_date_obj} has no ground truth; skipping")
        continue
    if model_name == 'gt':
        preds = gt.loc[target_date_obj].rename('pred').to_frame().reset_index()
    else:
        if not os.path.exists(file_path):
            printf(f"Warning: {file_path} does not exist; skipping")
            continue
        # Obtain a lock on the file to deal with multiple process file access
        with FileLock(file_path+"lock"):
             preds = pd.read_hdf(file_path)
        subprocess.call(f"rm {file_path}lock", shell=True)
    
    if len(preds) == 0:
        printf(f"There are no predictions in {file_path}; skipping")
        continue
    preds = preds.set_index(['lat','lon']).loc[:,'pred'].sort_index().astype('float64')
    assert len(preds) == len(gt.loc[target_date_obj]), f"Differing lengths for prediction ({len(preds)}) and ground truth ({len(gt.loc[target_date_obj])})"

    #printf('-Calculating metrics')
    if 'wtd_mse' in metrics:
        if weights is None:
            # Compute cosine weights for each latitude in order of appearance
            # Weights are the same for each target date so only need to compute once
            lats = gt.loc[target_date_obj].index.get_level_values('lat').values
            weights = np.cos(np.deg2rad(np.abs(lats)))
        metric_dfs['wtd_mse'].loc[target_date_obj] = get_wtd_mse(preds, gt.loc[target_date_obj], weights)
    if 'mse' in metrics:
        metric_dfs['mse'].loc[target_date_obj] = get_mse(preds, gt.loc[target_date_obj])
    if 'rmse' in metrics or 'score' in metrics:
        rmse = get_rmse(preds, gt.loc[target_date_obj])
        if 'rmse' in metrics:
            metric_dfs['rmse'].loc[target_date_obj] = rmse
        if 'score' in metrics:
            metric_dfs['score'].loc[target_date_obj] = mean_rmse_to_score(rmse)
    if 'anom' in metrics:
        month_day = (target_date_obj.month, target_date_obj.day)
        if month_day == (2,29):
            printf('--Using Feb. 28 climatology for Feb. 29')
            month_day = (2,28)
        anom = get_anom(preds, clim.loc[month_day])
        metric_dfs['anom'].loc[target_date_obj] = anom
    if 'error' in metrics:
        error = get_error(preds, gt.loc[target_date_obj])
        metric_dfs['error'].loc[target_date_obj] = error
    if 'skill' in metrics:
        month_day = (target_date_obj.month, target_date_obj.day)
        if month_day == (2,29):
            printf('--Using Feb. 28 climatology for Feb. 29')
            month_day = (2,28)
        metric_dfs['skill'].loc[target_date_obj] = get_skill(
            preds, gt.loc[target_date_obj], clim.loc[month_day])
    if 'lat_lon_rmse' in metrics:
        sqd_error = np.square(preds - gt.loc[target_date_obj])
        if num_dates == 0:
            metric_dfs['lat_lon_rmse'] = sqd_error
            metric_dfs['lat_lon_rmse'].name = 'lat_lon_rmse'
        else:
            metric_dfs['lat_lon_rmse'] += sqd_error
        num_dates += 1
    if 'lat_lon_error' in metrics:
        error = preds - gt.loc[target_date_obj]
        if num_dates_lle == 0:
            metric_dfs['lat_lon_error'] = error
            metric_dfs['lat_lon_error'].name = 'lat_lon_error'
        else:
            metric_dfs['lat_lon_error'] += error
        num_dates_lle += 1
    if 'lat_lon_skill' in metrics:
        month_day = (target_date_obj.month, target_date_obj.day)
        if month_day == (2,29):
            printf('--Using Feb. 28 climatology for Feb. 29')
            month_day = (2,28)
        if num_dates_lls ==0:
            metric_dfs['lat_lon_skill'] = pd.DataFrame(index=preds.index, columns=['lat_lon_skill'])
            lat_lon_skill_u, lat_lon_skill_v = pd.DataFrame(index=preds.index), pd.DataFrame(index=preds.index) 
        lat_lon_skill_u[target_date_str] = preds - clim.loc[month_day]
        lat_lon_skill_v[target_date_str] = gt.loc[target_date_obj] - clim.loc[month_day]
        num_dates_lls += 1    
    if 'lat_lon_anom' in metrics:
        month_day = (target_date_obj.month, target_date_obj.day)
        if month_day == (2,29):
            printf('--Using Feb. 28 climatology for Feb. 29')
            month_day = (2,28)
        anom = preds - clim.loc[month_day]
        if num_dates_lla == 0:
            metric_dfs['lat_lon_anom'] = anom
            metric_dfs['lat_lon_anom'].name = 'lat_lon_anom'
        else:
            metric_dfs['lat_lon_anom'] += anom
        num_dates_lla += 1    
    if 'lat_lon_pred' in metrics:
        if num_dates_llp == 0:
            metric_dfs['lat_lon_pred'] = preds
            metric_dfs['lat_lon_pred'].name = 'lat_lon_pred'
        else:
            metric_dfs['lat_lon_pred'] += preds
        num_dates_llp += 1    
    toc()
    

if 'lat_lon_pred' in metric_dfs:
    # Replace preds sum with mean preds
    metric_dfs['lat_lon_pred'] /= num_dates_llp
    metric_dfs['lat_lon_pred'] = metric_dfs['lat_lon_pred']  
    
if 'lat_lon_anom' in metric_dfs:
    # Replace preds sum with mean preds
    metric_dfs['lat_lon_anom'] /= num_dates_lla
    metric_dfs['lat_lon_anom'] = metric_dfs['lat_lon_anom']   
    
if 'lat_lon_skill' in metric_dfs:
    # Calculate skill between u and v vectors
    l = [(1 - cosine(lat_lon_skill_u.loc[i], lat_lon_skill_v.loc[i])) for i in lat_lon_skill_u.index]  
    metric_dfs['lat_lon_skill']['lat_lon_skill'] = l
    metric_dfs['lat_lon_skill'] = metric_dfs['lat_lon_skill'].squeeze()    

if 'lat_lon_rmse' in metric_dfs:
    # Replace error sum with RMSE
    metric_dfs['lat_lon_rmse'] /= num_dates
    metric_dfs['lat_lon_rmse'] = np.sqrt(metric_dfs['lat_lon_rmse'])
    
if 'lat_lon_error' in metric_dfs:
    # Replace error sum with RMSE
    metric_dfs['lat_lon_error'] /= num_dates_lle  

toc()

# Create output directory if it doesn't exist
make_directories(output_folder)

# Set error columns to float and print diagnostics
for metric, df in metric_dfs.items():
    printf(f'\n\n{metric}')
    printf(', '.join([f'{statistic}:{np.round(value, 3)}'
                      for statistic, value in df.describe()[1:].items()]))
    if metric == 'rmse':
        printf(f'- bonus score: {mean_rmse_to_score(df.mean())}')
printf('')

# Save error dfs
printf('')
for metric, df in metric_dfs.items():
    if df.isna().all():
        printf(f'{metric} dataframe is empty; not saving')
        continue
    metric_file_path = f'{output_folder}/{metric}-{gt_id}_{horizon}-{target_dates}.h5'
    pandas2hdf(df.reset_index(), metric_file_path, format='table')