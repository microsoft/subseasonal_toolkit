# Create batch forecasts using a linear ensemble for a specified test set
# dates. Output is stored in models/linear_ensemble/
#
# Example usage:
#   python src/models/linear_ensemble/batch_predict.py contest_tmp2m 34w -t std_test
#   python src/models/linear_ensemble/batch_predict.py contest_tmp2m 34w -t std_contest -m cfsv2pp,perppmclimpp 
#
# Positional args:
#   gt_id: contest_tmp2m or contest_precip
#   horizon: 34w or 56w
#
# Named args:
#   --model_names (-m): target models (and submodel, if applicable), to be specified separated by commas,
#                       e.g. spatiotemporal_mean:spatiotemporal_mean-1981_2016,lrr
#   --target_dates (-t): target dates for batch prediction (default: 'std_test')

import os
import importlib
import numpy as np
import pandas as pd
import pickle
import time
import warnings
from datetime import datetime
from argparse import ArgumentParser
from joblib import Parallel, delayed

from subseasonal_toolkit.utils.models_util import save_forecasts, get_forecast_filename
from subseasonal_toolkit.utils.eval_util import get_target_dates
from subseasonal_toolkit.utils.general_util import printf, set_file_permissions, make_directories, symlink
from subseasonal_toolkit.models.linear_ensemble.attributes import get_submodel_name, get_model_names

from subseasonal_data import data_loaders

from ensemble_models import LinearEnsemble, StepwiseFeatureSelectionWrapper, REQUIRED_COLS
from ensemble_utils import _ensure_permissions, _merge_dataframes, _get_model_file_path

def _get_dates_list(input_dates, horizon):
    if len(input_dates) == 1:
        input_dates_objs = get_target_dates(date_str=input_dates[0], horizon=horizon)
    else:
        input_dates_objs = np.concatenate(
            [get_target_dates(date_str=td, horizon=horizon) for td in input_dates])
    return input_dates_objs


# Get command args
parser = ArgumentParser(
    description='Command line arguments for model ensembling.')
parser.add_argument("pos_vars", nargs="*")  # gt_id and horizon
parser.add_argument('--model_names', '-m', default="tuned_localboosting,tuned_cfsv2pp,tuned_climpp,perpp,multillr,tuned_salient2",
                    help="Comma separated list of models e.g., 'climpp,cfsv2pp,perpp,multillr'")
parser.add_argument('-t', '--target_dates', type=str, nargs='+', default=["std_val", "std_test"],
                    help="Dates to predict on")
parser.add_argument('--forecast', '-f', default=None, 
                        help="include the forecasts of this dynamical model as features")

# Assign arguments to variables
args, opt = parser.parse_known_args()
gt_id = args.pos_vars[0]  # "contest_precip" or "contest_tmp2m"
horizon = args.pos_vars[1]  # "34w" or "56w"
model_string = args.model_names
forecast = args.forecast

# Perpare input models
if forecast is not None:
    model_string = get_model_names(forecast, horizon=horizon)
models = model_string.split(',')
models.sort()
model_string = (",").join(models)
if len(models) == 1:
    raise ValueError(
        "At least 2 models must be provided for ensembling and only 1 was provided as input.")
target_dates = args.target_dates

# Get target dates
target_dates_objs = _get_dates_list(target_dates, horizon)

# Create abc model
if forecast is not None:
    # Record output model name and submodel name
    output_model_name = f"abc_{forecast}"
    submodel_name = get_submodel_name(model_names=model_string)
    # Create directory for storing forecasts if one does not already exist
    out_dir = os.path.join("models", output_model_name, "submodel_forecasts", 
                           submodel_name, f"{gt_id}_{horizon}")
    if not os.path.exists(out_dir):
        make_directories(out_dir)    
    
# File path template
submodel_fname_template = os.path.join(
    "models","{model_name}","submodel_forecasts","{submodel_name}",
    "{gt_id}_{horizon}","{gt_id}_{horizon}-{target_date}.h5")

target_fnames = {model: [_get_model_file_path(
    model, submodel_fname_template, gt_id,
    horizon, target_date) for target_date in target_dates_objs] for model in models}

# No training necessary
# Reading the serialized model and predicting
submodel_name = get_submodel_name(model_names=model_string)
output_dir = os.path.join("models","linear_ensemble","submodel_forecasts",
                          "{submodel_name}","{gt_id}_{horizon}").format(
                              submodel_name=submodel_name,
                              gt_id=gt_id,
                              horizon=horizon)
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory is {output_dir}")
model_path = os.path.join(output_dir, "learned_ensemble_model.pkl")
if not os.path.exists(model_path):
    # The model is just a simple average and does not require training
    # Always default to the pickle file in case some of the models were missing at train time
    linear_ensemble_model = LinearEnsemble(local=False, dynamic=False)
    linear_ensemble_model.fit(pd.DataFrame(
        columns=REQUIRED_COLS + [f"pred_{this_model}" for this_model in models]))
    pickle.dump(linear_ensemble_model, open(model_path, 'wb'))
    set_file_permissions(model_path)
else:
    linear_ensemble_model = pickle.load(open(model_path, 'rb'))

# Merge test dataframes in a start_date, lat,lon, model1, model2,..., modeln dataframes
# And apply models from above step
print("Predicting on target dates...", end='')
start = time.time()    
test_dataframe = _merge_dataframes(models, target_fnames)

def _target_predict(target_date):
    target_test_dataframe = test_dataframe[test_dataframe['start_date'] == target_date]
    preds = target_test_dataframe[["lat", "lon", "start_date"]].copy()
    target_test_dataframe = target_test_dataframe.dropna()
    target_date_str = datetime.strftime(target_date, '%Y%m%d')
    if target_test_dataframe.empty:
        print(f"Missing date {target_date_str}")
#         warnings.warn("The dataframe for target date {} is empty. Please check the model outputs and rerun this script.".format(target_date_str))
        preds["pred"] = np.nan
    else:
        preds = linear_ensemble_model.predict(target_test_dataframe)
        # Write results to file
        save_forecasts(
            preds,
            model="linear_ensemble", submodel=submodel_name,
            gt_id=gt_id, horizon=horizon,
            target_date_str=target_date_str)
        if forecast is not None:
            src_file = get_forecast_filename(model="linear_ensemble", submodel=submodel_name,
                                             gt_id=gt_id, horizon=horizon, target_date_str=target_date_str)
            dst_file = get_forecast_filename(model=f"abc_{forecast}", submodel=submodel_name,
                                             gt_id=gt_id, horizon=horizon, target_date_str=target_date_str)
            symlink(src_file, dst_file, use_abs_path=True)
    
 
# Parallelize target predictions
Parallel(n_jobs=-1, verbose=1, backend='threading')(
    delayed(_target_predict)(target_date) for target_date in target_dates_objs)
end = time.time()
print("Done in {0:.2f}s\n".format(end-start))
