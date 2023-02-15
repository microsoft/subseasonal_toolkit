# tuned_ecmwfpp Shift
# Apply tuned_ecmwfpp correction learned on model ensemble to individual ensemble members
#
# Example usage:
#   python -m subseasonal_toolkit.models.tuned_ecmwfpps.batch_predict us_tmp2m_1.5x1.5 34w -t std_paper_forecast -y 3 -m None -fw c
#
# Positional args:
#   gt_id: contest_tmp2m, contest_precip, us_tmp2m, or us_precip
#   horizon: 34w or 56w
#
# Named args:
#   --target_dates (-t): target dates for batch prediction
#   --num_years (-y): number of years to use in tuning ("all" for all years
#     or positive integer); (default: "all")
#   --margin_in_days (-m): number of month-day combinations on either side of 
#     the target combination to include; set to 0 to include only target 
#     month-day combo; set to None to include entire year; (default: None)
#   --forecast_with (-fw): Generate forecast using the control (c),
#     average perturbed (p), single perturbed (p1, ..., p50), 
#     or perturbed-control ensemble (p+c) ECMWF forecast; (default: "p+c")
#   --debias_with (-dw): Debias using the control (c), average perturbed (p), 
#     or perturbed-control ensemble (p+c) ECMWF reforecast; (default: "p+c")

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
model_name = "tuned_ecmwfpps"

parser = ArgumentParser()
parser.add_argument("pos_vars",nargs="*")  # gt_id and horizon 
parser.add_argument('--target_dates', '-t', default="std_test")
parser.add_argument('--num_years', '-y', default="all",
                   help="Number of years to use in training (all or integer)")
parser.add_argument('--margin_in_days', '-m', default="None", 
                   help="Number of month-day combinations on either side of the target combination to include; "
                        "set to 0 to include only target month-day combo; set to None to include entire year; "
                        "None by default")
parser.add_argument('--forecast_with', '-fw', default="p+c", 
                    help="Generate forecast using the control (c), "
                    "average perturbed (p), single perturbed (p1, ..., p50), "
                    "or perturbed-control ensemble (p+c) ECMWF forecast.")
parser.add_argument('--debias_with', '-dw', default="p+c", 
                    help="Debias using the control (c), average perturbed (p), "
                    "or perturbed-control ensemble (p+c) ECMWF reforecast.")  
args = parser.parse_args()

# Assign variables
gt_id = args.pos_vars[0] # e.g., "contest_precip" or "contest_tmp2m"
horizon = args.pos_vars[1] # e.g., "12w", "34w", or "56w"    
target_dates = args.target_dates
num_years = args.num_years
if num_years != "all":
    num_years = int(num_years)
margin_in_days = args.margin_in_days
if margin_in_days == "None":
    margin_in_days = None
else:
    margin_in_days = int(args.margin_in_days)
debias_with = args.debias_with
forecast_with = args.forecast_with 

# Get submodel names for tuned_ecmwfpps, tuned_ecmwfpp, and raw_ecmwf
sn_abcs = get_submodel_name(
    model_name, num_years=num_years,
    margin_in_days=margin_in_days,
    debias_with=debias_with,
    forecast_with=forecast_with)
print(f"tuned_ecmwfpps submodel: {sn_abcs}")
sn_abc_selected = get_selected_submodel_name(
    model="tuned_ecmwfpp", gt_id=gt_id, horizon=horizon)
print(f"selected tuned_ecmwfpp submodel: {sn_abc_selected}")
sn_raw_selected = get_selected_submodel_name(
    model="raw_ecmwf", gt_id=gt_id, horizon=horizon)
print(f"selected raw_ecmwf submodel: {sn_raw_selected}")
if horizon == "12w":
    first_lead, last_lead = (1,1)
elif horizon == "34w":
    first_lead, last_lead = (15,15)
elif horizon == "56w":
    first_lead, last_lead = (29,29)
sn_raw = get_submodel_name(
    "raw_ecmwf", first_lead=first_lead, last_lead = last_lead, 
    forecast_with=forecast_with)
print(f"raw_ecmwf submodel: {sn_raw}")

# Get forecasting task
task = f"{gt_id}_{horizon}"

# Get list of target date objects
target_date_objs = pd.Series(get_target_dates(date_str=target_dates,horizon=horizon))

# Generate predictions
for target_date_obj in target_date_objs:
        
    target_date_str = datetime.strftime(target_date_obj, '%Y%m%d')
    
    # Skip if forecast already produced for this target
    forecast_file = get_forecast_filename(
        model=model_name, submodel=sn_abcs, 
        gt_id=gt_id, horizon=horizon, 
        target_date_str=target_date_str)
    
    if True and os.path.isfile(forecast_file):
        printf(f"prior forecast exists for target={target_date_obj}; skipping")
        continue
    
    printf(f'target={target_date_str}')
    tic()

    # Load raw forecast
    forecast_file = get_forecast_filename(
        model="raw_ecmwf", submodel=sn_raw, 
        gt_id=gt_id, horizon=horizon, 
        target_date_str=target_date_str)
    if not os.path.isfile(forecast_file):
        printf(f"raw forecast missing for target={target_date_obj}; skipping")
        continue
    preds = pd.read_hdf(forecast_file).set_index(['lat','lon','start_date'])
    
    # Adjust raw forecast with tuned_ecmwfpp shift = 
    # selected tuned_ecmwfpp forecast - selected raw forecast
    forecast_file = get_forecast_filename(
        model="tuned_ecmwfpp", submodel=sn_abc_selected, 
        gt_id=gt_id, horizon=horizon, 
        target_date_str=target_date_str)
    if not os.path.isfile(forecast_file):
        printf(f"selected abc forecast missing for target={target_date_obj}; skipping")
        continue
    preds += pd.read_hdf(forecast_file).set_index(['lat','lon','start_date'])
    forecast_file = get_forecast_filename(
        model="raw_ecmwf", submodel=sn_raw_selected, 
        gt_id=gt_id, horizon=horizon, 
        target_date_str=target_date_str)
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
        model=model_name, submodel=sn_abcs, 
        gt_id=gt_id, horizon=horizon, 
        target_date_str=target_date_str)
    toc()
    
