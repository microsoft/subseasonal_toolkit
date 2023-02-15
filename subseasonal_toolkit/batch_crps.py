# Evaluate crps and lat_lon_crps error metrics for model for a specified set of test
# dates. Output is stored in eval/metrics/MODEL_NAME/. 
#
# Example usage:
#   python -m subseasonal_toolkit.batch_crps us_precip_1.5x1.5 34w -mn deb_ecmwf -t std_paper_forecast
#   python -m subseasonal_toolkit.batch_crps us_precip_1.5x1.5 34w -mn abcds_ecmwf -t std_paper_forecast
#
# Positional args:
#   gt_id: e.g., contest_tmp2m, contest_precip, us_tmp2m, us_precip
#   horizon: 12w, 34w, or 56w
#
# Named args:
#   --target_dates (-t): target dates for batch prediction (e.g.,
#     'std_val','std_paper','std_ens') (default: 'std_paper')
#   --model_name (-mn): name of model, e.g, d2p_deb_ecmwf (default: None)

import os
from os.path import isfile
import subprocess
from datetime import datetime
import numpy as np
import pandas as pd
import properscoring as ps
from argparse import ArgumentParser
from subseasonal_data.utils import get_measurement_variable
from subseasonal_data.data_loaders import get_ground_truth
from subseasonal_toolkit.utils.general_util import printf, make_directories, tic, toc
from subseasonal_toolkit.utils.experiments_util import pandas2hdf
from subseasonal_toolkit.utils.eval_util import (get_target_dates, 
                                                 get_task_metrics_dir)
from subseasonal_toolkit.utils.models_util import (
    get_task_forecast_dir,  get_d2p_submodel_names, get_forecast_filename)

# Load command line arguments
parser = ArgumentParser()
parser.add_argument("pos_vars", nargs="*")  # gt_id and horizon
parser.add_argument('--target_dates', '-t', default='std_paper')
parser.add_argument('--model_name', '-mn', default=None)
args = parser.parse_args()

# Assign variables
gt_id = args.pos_vars[0]  # e.g., "contest_precip" or "contest_tmp2m"
horizon = args.pos_vars[1]  # e.g., "12w", "34w", or "56w"
target_dates = args.target_dates
metrics = ["crps", "lat_lon_crps"]
model_name = args.model_name

# Set input folder (with pred files) and output folder (for metrics)
preds_folder = get_task_forecast_dir(
    model=model_name, gt_id=gt_id, horizon=horizon,
    target_dates=target_dates)
output_folder = get_task_metrics_dir(
    model=model_name, gt_id=gt_id, horizon=horizon,
    target_dates=target_dates)

# Get preds filenames
printf('Getting target dates')
tic()
# Use set of test dates to determine which preds dfs to load
target_date_objs = get_target_dates(date_str=target_dates, horizon=horizon)
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
    if metric == 'lat_lon_crps':
        # Keep track of number of dates contributing to error calculation
        # Initialize dataframe later
        num_dates_llc = 0
        continue
    # Index by target dates
    metric_dfs[metric] = pd.Series(name=metric, index=target_date_objs, dtype=np.float64)
    metric_dfs[metric].index.name = 'start_date'

# Get list of deterministic submodels to ensemble in forming
# probabilistic forecasts
det_submodel_names = get_d2p_submodel_names(model_name, gt_id, horizon)

# Get forecast file template for each deterministic submodel
det_templates = [os.path.join(
    get_task_forecast_dir(
        model=model_name,
        submodel=det_submodel_name,
        gt_id=gt_id,
        horizon=horizon), 
    f"{gt_id}_{horizon}"+"-{}.h5") 
    for det_submodel_name in det_submodel_names]

tic()
for target_date_obj in target_date_objs:
    printf(f'Getting metrics for {target_date_obj}')
    target_date_str = datetime.strftime(target_date_obj, '%Y%m%d')
    printf(target_date_str)
    tic()
    if target_date_obj not in gt.index:
        printf(f"Warning: {target_date_obj} has no ground truth; skipping")
        continue
    
    # Load each deterministic forecast for this target date
    det_forecasts = []
    for det_template in det_templates:
        # Load deterministic forecast
        det_forecast_file = det_template.format(target_date_str)
        if isfile(det_forecast_file):
            det_forecasts.append(
                pd.read_hdf(det_forecast_file).loc[:,['lat','lon','pred']].set_index(['lat','lon']).squeeze())
        
    if not det_forecasts:
        printf(f"\nno deterministic forecasts for target={target_date_obj}; skipping")
        toc()
        continue
    if len(det_forecasts) == 1:
        printf(f"\nonly one deterministic forecast for target={target_date_obj}; skipping")
        toc()
        continue
    # Form deterministic forecast dataframe
    det_forecasts = pd.concat(det_forecasts, axis=1).astype('float')
    
    # Compute CRPS
    crps = ps.crps_ensemble(gt.loc[target_date_obj].values, det_forecasts.values)
    
    # Store spatial average of CRPS
    metric_dfs['crps'].loc[target_date_obj] = crps.mean()
    
    # Store temporal sum of CRPS
    if num_dates_llc == 0:
        metric_dfs['lat_lon_crps'] = pd.Series(data=crps, index=det_forecasts.index)
        metric_dfs['lat_lon_crps'].name = 'lat_lon_crps'
    else:
        metric_dfs['lat_lon_crps'] += crps
    num_dates_llc += 1
    toc()
    
# Replace error sum with average
metric_dfs['lat_lon_crps'] /= num_dates_llc
toc()

# Create output directory if it doesn't exist
make_directories(output_folder)

# Set error columns to float and print diagnostics
for metric, df in metric_dfs.items():
    printf(f'\n\n{metric}')
    printf(', '.join([f'{statistic}:{np.round(value, 3)}'
                      for statistic, value in df.describe()[1:].items()]))
printf('')

# Save error dfs
printf('')
for metric, df in metric_dfs.items():
    if df.isna().all():
        printf(f'{metric} dataframe is empty; not saving')
        continue
    metric_file_path = f'{output_folder}/{metric}-{gt_id}_{horizon}-{target_dates}.h5'
    pandas2hdf(df.reset_index(), metric_file_path, format='table')