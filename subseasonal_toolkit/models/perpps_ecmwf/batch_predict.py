# Persistence++-ECMWF Shift
# Apply Persistence++ correction learned on model ensemble to individual ensemble members
#
# Example usage:
#   python -m subseasonal_toolkit.models.perpps_ecmwf.batch_predict us_tmp2m_1.5x1.5 34w -t std_paper_forecast -v c
#
# Positional args:
#   gt_id: contest_tmp2m, contest_precip, us_tmp2m, or us_precip
#   horizon: 34w or 56w
#
# Named args:
#   --target_dates (-t): target dates for batch prediction
#   --train_years (-y): number of years to use in training ("all" for all years
#     or positive integer); (default: "all")
#   --margin_in_days (-m): number of month-day combinations on either side of 
#     the target combination to include; set to 0 to include only target 
#     month-day combo; set to "None" to include entire year; (default: "None")
#   --version (-v): Which version of the ECMWF forecasts to use when training;
#     valid choices include cf (for control forecast), 
#     pf (for average perturbed forecast), ef (for control+perturbed ensemble),
#     or pf1, ..., pf50 for a single perturbed forecast

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
from subseasonal_toolkit.models.linear_ensemble.attributes import get_model_names

#
# Specify model parameters
#
model_name = "perpps_ecmwf"

parser = ArgumentParser()
parser.add_argument("pos_vars",nargs="*")  # gt_id and horizon                                                                                  
parser.add_argument('--target_dates', '-t', default="std_contest")

# Number of years to use in training ("all" or integer)
parser.add_argument('--train_years', '-y', default="all")

# Number of month-day combinations on either side of the target combination 
# to include when training
# Set to 0 to include only target month-day combo
# Set to "None" to include entire year
parser.add_argument('--margin_in_days', '-m', default="None")

# Which version of the ECMWF forecasts to use when training
# Valid choices include cf (for control forecast), 
# pf (for average perturbed forecast), ef (for control+perturbed ensemble),
# or pf1, ..., pf50 for a single perturbed forecast
parser.add_argument('--version', '-v', default="ef")

args, opt = parser.parse_known_args()

# Assign variables        
gt_id = args.pos_vars[0] # "contest_precip" or "contest_tmp2m"                                                                            
horizon = args.pos_vars[1] # "34w" or "56w"                                                                                        
target_dates = args.target_dates
train_years = args.train_years
if train_years != "all":
    train_years = int(train_years)
if args.margin_in_days == "None":
    margin_in_days = None
else:
    margin_in_days = int(args.margin_in_days)
version = args.version

# Get submodel names for perpps_ecmwf, perpp_ecmwf, and raw_ecmwf
sn_perpps = get_submodel_name(
    model_name, train_years=train_years, 
    margin_in_days=margin_in_days, version=version)
printf(f"perpps_ecmwf submodel: {sn_perpps}")
sn_perpp_selected = get_selected_submodel_name(
    model="perpp_ecmwf", gt_id=gt_id, horizon=horizon)
printf(f"selected perpp_ecmwf submodel: {sn_perpp_selected}")
sn_raw_selected = get_selected_submodel_name(
    model="raw_ecmwf", gt_id=gt_id, horizon=horizon)
printf(f"selected raw_ecmwf submodel: {sn_raw_selected}")
if horizon == "12w":
    first_lead, last_lead = (1,1)
elif horizon == "34w":
    first_lead, last_lead = (15,15)
elif horizon == "56w":
    first_lead, last_lead = (29,29)
if version == "ef":
    forecast_with = "p+c"
else:
    # Remove the letter "f" from version
    forecast_with = version[:1]+version[2:]
sn_raw = get_submodel_name(
    "raw_ecmwf", first_lead=first_lead, last_lead = last_lead, 
    forecast_with=forecast_with)
printf(f"raw_ecmwf submodel: {sn_raw}")

# Get forecasting task
task = f"{gt_id}_{horizon}"

# Get list of target date objects
target_date_objs = pd.Series(get_target_dates(date_str=target_dates,horizon=horizon))

# Generate predictions
for target_date_obj in target_date_objs:
        
    target_date_str = datetime.strftime(target_date_obj, '%Y%m%d')
    
    # Skip if forecast already produced for this target
    forecast_file = get_forecast_filename(
        model=model_name, submodel=sn_perpps, 
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
    
    # Adjust raw forecast with perpp shift = 
    # selected perpp forecast - selected raw forecast
    forecast_file = get_forecast_filename(
        model="perpp_ecmwf", submodel=sn_perpp_selected, 
        gt_id=gt_id, horizon=horizon, 
        target_date_str=target_date_str)
    if not os.path.isfile(forecast_file):
        printf(f"selected perpp forecast missing for target={target_date_obj}; skipping")
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
        model=model_name, submodel=sn_perpps, 
        gt_id=gt_id, horizon=horizon, 
        target_date_str=target_date_str)
    toc()
    
