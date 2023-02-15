# Predicts outcomes using raw (undebiased) ecmwf ensemble forecast
#
# Example usage:
#   python -m subseasonal_toolkit.models.raw_ecmwf.batch_predict us_tmp2m_1.5x1.5 34w -t std_paper_forecast -fl 15 -ll 15 -fw c
#
# Positional args:
#   gt_id: e.g., contest_tmp2m, contest_precip, us_tmp2m, or us_precip
#   horizon: 12w, 34w or 56w
#
# Named args:
#   --target_dates (-t): target dates for batch prediction (default: std_test)
#
# Note: Other named arguments are supported but should not be set
import os
import json
import subprocess
import shlex
import sys
from argparse import ArgumentParser
from pathlib import Path
from subseasonal_toolkit.utils.models_util import get_submodel_name, get_task_forecast_dir
from subseasonal_toolkit.utils.general_util import printf, make_parent_directories, symlink
from pkg_resources import resource_filename

model_name = "raw_ecmwf"
base_model_name = "ecmwfpp"

# Load command line arguments
parser = ArgumentParser()
parser.add_argument("pos_vars",nargs="*")  # gt_id and horizon
parser.add_argument('--target_dates', '-t', default="std_paper_forecast")
parser.add_argument('--first_lead', '-fl', default=1, 
                    help="first ecmwf lead to average into forecast (0-29)")
parser.add_argument('--last_lead', '-ll', default=1, 
                    help="last ecmwf lead to average into forecast (0-29)")
parser.add_argument('--forecast_with', '-fw', default="c", 
                        help="Generate forecast using the perterbed (p) or control (c) ECMWF forecast (or p+c for both).") 
args = parser.parse_args()

# Assign variables
gt_id = args.pos_vars[0] # e.g., "contest_precip" or "contest_tmp2m"
horizon = args.pos_vars[1] # e.g., "34w" or "56w"

# Create dictionary of default values for additional arguments
# used by the base model
default_args = {'fit_intercept': False,
                'train_years': 20, 
                'margin_in_days': 0,
                'first_day': 1,
                'last_day': 1,
                'loss': "mse",
                'debias_with': "p+c"}
default_arg_str = " ".join(f"--{key} {value}" for key, value in default_args.items())

# Reconstruct command-line arguments and run base model
cmd_args = " ".join(map(shlex.quote, sys.argv[1:]))
cmd_args = f"{cmd_args} {default_arg_str}"
predict_script = resource_filename("subseasonal_toolkit", os.path.join("models",base_model_name,"batch_predict.py"))
cmd = f"python {predict_script} {cmd_args}"

printf(f"Running {cmd}")
subprocess.call(cmd, shell=True)

# Remove target dates and positional arguments from args dictionary
# so that remaining arguments are model-specific
del args.pos_vars
del args.target_dates

# Soft link prediction directory for this model to relevant base model 
# prediction directory
base_submodel_name = get_submodel_name(
    model=base_model_name, **default_args, **vars(args))
base_submodel_dir = get_task_forecast_dir(
    model=base_model_name, submodel=base_submodel_name, 
    gt_id=gt_id, horizon=horizon)
# Same submodel name used for model and base model
submodel_dir = get_task_forecast_dir(
    model=model_name, submodel=base_submodel_name, 
    gt_id=gt_id, horizon=horizon)

# Ensure all parent directories of submodel dir exist with correct permissions
make_parent_directories(submodel_dir)

printf(f"Soft-linking\n-src: {base_submodel_dir}\n-dest: {submodel_dir}")
symlink(base_submodel_dir, submodel_dir, use_abs_path=True)
