# shift_deb_loess_ecmwf
# Apply deb_loess_ecmwf correction learned on model ensemble to individual ensemble members
#
# Example usage:
#   python -m subseasonal_toolkit.models.shift_deb_loess_ecmwf.batch_predict us_tmp2m_1.5x1.5 34w -t std_paper_forecast -y 3 -m None -fw c
#
# Positional args:
#   gt_id: contest_tmp2m, contest_precip, us_tmp2m, or us_precip
#   horizon: 34w or 56w
#
# Named args:
#   --target_dates (-t): target dates for batch prediction
#   --num_years (-y): number of years to use in tuning ("all" for all years
#     or positive integer); (default: "all")
#   --forecast_with (-fw): Generate forecast using the control (c),
#     average perturbed (p), single perturbed (p1, ..., p50), 
#     or perturbed-control ensemble (p+c) ECMWF forecast; (default: "p+c")
#   --loess_frac (-lf): Fraction, in [0, 1], of data used for loess; 
#     smaller means less smoothing

import os
from pkg_resources import resource_filename
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import sys
from datetime import datetime
from subseasonal_toolkit.utils.general_util import printf, tic, toc
from subseasonal_toolkit.utils.eval_util import get_target_dates
from subseasonal_toolkit.utils.models_util import (
    get_submodel_name, get_selected_submodel_name, get_forecast_filename, 
    save_forecasts)

#
# Specify model parameters
#
model_name = "shift_deb_loess_ecmwf"

parser = ArgumentParser()
parser.add_argument("pos_vars",nargs="*")  # gt_id and horizon 
parser.add_argument('--target_dates', '-t', default="std_test")
parser.add_argument('--forecast_with', '-fw', default="p+c", 
                    help="Generate forecast using the control (c) "
                    "or single perturbed (p1, ..., p50) ECMWF forecast.")
parser.add_argument(
    "--loess_frac", "-lf", default=0.1, type=float, 
    help="Fraction, in [0, 1], of data used for loess; smaller means less smoothing"
)
args = parser.parse_args()

# Assign variables
gt_id = args.pos_vars[0] # e.g., "contest_precip" or "contest_tmp2m"
horizon = args.pos_vars[1] # e.g., "12w", "34w", or "56w"    
target_dates = args.target_dates
forecast_with = args.forecast_with 
loess_frac = args.loess_frac

# Get file name templates for shifted, raw, and baseline models
sn_shift = get_submodel_name(
    model_name, 
    forecast_with=forecast_with,
    loess_frac=loess_frac)
template_shift = get_forecast_filename(
        model=model_name, submodel=sn_shift, 
        gt_id=gt_id, horizon=horizon, 
        target_date_str="{}")
print(f"shift_deb_loess_ecmwf template: {template_shift}")


# sn_loess_selected = get_selected_submodel_name(
#     model="deb_loess_ecmwf", gt_id=gt_id, horizon=horizon)
template_loess_selected = get_forecast_filename(
        model="deb_loess_ecmwf", 
        gt_id=gt_id, horizon=horizon, 
        target_date_str="{}")
print(f"selected deb_loess_ecmwf template: {template_loess_selected}")

# sn_raw_selected = get_selected_submodel_name(
#     model="raw_ecmwf", gt_id=gt_id, horizon=horizon)
template_raw_selected = get_forecast_filename(
        model="raw_ecmwf", 
        gt_id=gt_id, horizon=horizon, 
        target_date_str="{}")
print(f"selected raw_ecmwf template: {template_raw_selected}")

if horizon == "12w":
    first_lead, last_lead = (1,1)
elif horizon == "34w":
    first_lead, last_lead = (15,15)
elif horizon == "56w":
    first_lead, last_lead = (29,29)
sn_raw = get_submodel_name(
    "raw_ecmwf", first_lead=first_lead, last_lead = last_lead, 
    forecast_with=forecast_with)
template_raw = get_forecast_filename(
        model="raw_ecmwf", submodel=sn_raw, 
        gt_id=gt_id, horizon=horizon, 
        target_date_str="{}")
print(f"raw_ecmwf template: {template_raw}")

# Get forecasting task
task = f"{gt_id}_{horizon}"

# Get list of target date objects
target_date_objs = pd.Series(get_target_dates(date_str=target_dates,horizon=horizon))

# Generate predictions
for target_date_obj in target_date_objs:
        
    target_date_str = datetime.strftime(target_date_obj, '%Y%m%d')
    
    # Skip if forecast already produced for this target
    forecast_file = template_shift.format(target_date_str)
    
    if True and os.path.isfile(forecast_file):
        printf(f"prior forecast exists for target={target_date_obj}; skipping")
        continue
    
    printf(f'target={target_date_str}')
    tic()

    # Load raw forecast
    forecast_file = template_raw.format(target_date_str)
    if not os.path.isfile(forecast_file):
        printf(f"raw forecast missing for target={target_date_obj}; skipping")
        continue
    preds = pd.read_hdf(forecast_file).set_index(['lat','lon','start_date'])
    
    # Adjust raw forecast with deb_loess_ecmwf shift = 
    # selected deb_loess_ecmwf forecast - selected raw forecast
    forecast_file = template_loess_selected.format(target_date_str)
    if not os.path.isfile(forecast_file):
        printf(f"selected abc forecast missing for target={target_date_obj}; skipping")
        continue
    preds += pd.read_hdf(forecast_file).set_index(['lat','lon','start_date'])
    forecast_file = template_raw_selected.format(target_date_str)
    if not os.path.isfile(forecast_file):
        printf(f"selected raw forecast missing for target={target_date_obj}; skipping")
        continue
    preds -= pd.read_hdf(forecast_file).set_index(['lat','lon','start_date'])
    
    # Ensure raw precipitation predictions are never less than zero
    if gt_id.endswith("precip") or gt_id.endswith("precip_1.5x1.5"):
        tic()
        preds = np.maximum(preds,0)
        toc()
    
    # Save prediction to file in standard format
    save_forecasts(preds.reset_index(),
        model=model_name, submodel=sn_shift, 
        gt_id=gt_id, horizon=horizon, 
        target_date_str=target_date_str)
    toc()
    
