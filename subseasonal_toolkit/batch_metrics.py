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
#   --metrics (-m): Space-separated list of error metrics to compute (valid choices are rmse, score, skill, lat_lon_rmse);
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
from ttictoc import tic, toc
from subseasonal_data.utils import get_measurement_variable
from subseasonal_data.data_loaders import get_ground_truth, get_climatology
from subseasonal_toolkit.utils.general_util import printf, make_directories
from subseasonal_toolkit.utils.experiments_util import get_id_name, get_th_name, pandas2hdf
from subseasonal_toolkit.utils.eval_util import (get_target_dates, mean_rmse_to_score, 
                                                 get_task_metrics_dir)
from subseasonal_toolkit.utils.models_util import get_task_forecast_dir

# Load command line arguments
parser = ArgumentParser()
parser.add_argument("pos_vars", nargs="*")  # gt_id and horizon
parser.add_argument('--target_dates', '-t', default='std_paper')
# For metrics, 1 or more values expected => creates a list
parser.add_argument("--metrics", '-m', nargs="+", type=str, default=['rmse', 'score', 'skill'],
                    help="Space-separated list of error metrics to compute (valid choices are rmse, score, skill, lat_lon_rmse)")
parser.add_argument('--model_name', '-mn', default=None)
parser.add_argument('--submodel_name', '-sn', default=None)
args = parser.parse_args()

# Assign variables
gt_id = get_id_name(args.pos_vars[0])  # "contest_precip" or "contest_tmp2m"
horizon = get_th_name(args.pos_vars[1])  # "34w" or "56w"
target_dates = args.target_dates
metrics = args.metrics
model_name = args.model_name
submodel_name = args.submodel_name

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
    if metric == 'skill':
        # Load climatology
        printf('Loading climatology and replacing start date with month-day')
        tic()
        clim = get_climatology(gt_id)
        clim = clim.set_index(
            [clim.start_date.dt.month,clim.start_date.dt.day,'lat','lon']
        ).drop(columns='start_date').squeeze().sort_index()
        toc()

# Fill the error dfs for given target date
def get_rmse(pred, gt):
    return np.sqrt(np.square(pred-gt).mean())

def get_skill(pred, gt, clim):
    return 1 - cosine(pred-clim, gt-clim)

tic()
for file_path, target_date_obj in zip(file_paths, target_date_objs):
    printf(f'Getting metrics for {target_date_obj}')
    tic()
    if target_date_obj not in gt.index:
        printf(f"Warning: {target_date_obj} has no ground truth; skipping")
        continue
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
    if 'rmse' in metrics or 'score' in metrics:
        rmse = get_rmse(preds, gt.loc[target_date_obj])
        if 'rmse' in metrics:
            metric_dfs['rmse'].loc[target_date_obj] = rmse
        if 'score' in metrics:
            metric_dfs['score'].loc[target_date_obj] = mean_rmse_to_score(rmse)
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
    toc()

if 'lat_lon_rmse' in metric_dfs:
    # Replace error sum with RMSE
    metric_dfs['lat_lon_rmse'] /= num_dates
    metric_dfs['lat_lon_rmse'] = np.sqrt(metric_dfs['lat_lon_rmse'])
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
