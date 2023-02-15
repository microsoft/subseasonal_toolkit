# Predicts outcomes using subx_mean++
#
# Example usage:
#   python models/subx_meanpp/batch_predict.py us_tmp2m_1.5x1.5 34w -t std_paper_forecast -y all -m None
#   python models/subx_meanpp/batch_predict.py us_precip_1.5x1.5 34w -t std_paper_forecast -y 20 -m 56
#
# Positional args:
#   gt_id: us_tmp2m_1.5x1.5 or us_precip_1.5x1.5
#   horizon: 34w or 56w
#
# Named args:
#   --target_dates (-t): target dates for batch prediction
#   --train_years (-y): number of years to use in training ("all" for all years
#     or positive integer); (default: "all")
#   --margin_in_days (-m): number of month-day combinations on either side of 
#     the target combination to include; set to 0 to include only target 
#     month-day combo; set to "None" to include entire year; (default: "None")
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

forecast = "subx_mean"
model_name = f"{forecast}pp"
base_model_name = "subxpp"

# Load command line arguments
parser = ArgumentParser()
parser.add_argument("pos_vars",nargs="*")  # gt_id and horizon
parser.add_argument('--target_dates', '-t', default="std_contest")
parser.add_argument('--fit_intercept', '-i', default="False",
                    choices=['True', 'False'],
                    help="Fit intercept parameter if \"True\"; do not if \"False\"")
parser.add_argument('--train_years', '-y', default="all",
                    help="Number of years to use in training (\"all\" or integer)")
parser.add_argument('--margin_in_days', '-m', default="None",
                    help="number of month-day combinations on either side of the target combination "
                    "to include when training; set to 0 include only target month-day combo; "
                    "set to None to include entire year")
parser.add_argument('--first_day', '-fd', default=1,
                    help="first available daily subx forecast (1 or greater) to average")
parser.add_argument('--last_day', '-ld', default=1,
                    help="last available daily subx forecast (first_day or greater) to average")
parser.add_argument('--loss', '-l', default="mse", 
                    help="loss function: mse, rmse, skill, or ssm")
parser.add_argument('--first_lead', '-fl', default=0, 
                    help="first subx lead to average into forecast (0-29)")
parser.add_argument('--last_lead', '-ll', default=29, 
                    help="last subx lead to average into forecast (0-29)")
parser.add_argument('--mei', default=False, action='store_true', help="Whether to condition on MEI")
parser.add_argument('--mjo', default=False, action='store_true', help="Whether to condition on MJO")
args, opt = parser.parse_known_args()

# Add forecast argument
args.forecast = forecast

# Assign variables
gt_id = args.pos_vars[0] # "contest_precip" or "contest_tmp2m"
horizon = args.pos_vars[1] # "34w" or "56w"

# Reconstruct command-line arguments and run base model
cmd_args = " ".join(map(shlex.quote, sys.argv[1:]))
predict_script = resource_filename("subseasonal_toolkit", os.path.join("models",base_model_name,"batch_predict.py"))
cmd = f"python {predict_script} {cmd_args} --forecast {args.forecast}"

printf(f"Running {cmd}")
subprocess.call(cmd, shell=True)

# Remove target dates and positional arguments from args dictionary
# so that remaining arguments are model-specific
del args.pos_vars
del args.target_dates

# Soft link prediction directory for this model to relevant submodel 
# prediction directory
base_submodel_name = get_submodel_name(
    model=base_model_name, **vars(args))
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





