import os
import subprocess
import shlex
import sys
from argparse import ArgumentParser
from subseasonal_toolkit.utils.models_util import get_submodel_name, get_task_forecast_dir
from subseasonal_toolkit.utils.general_util import printf, make_parent_directories, symlink
from pkg_resources import resource_filename

forecast = "ecmwf"
model_name = f"deb_quantile_{forecast}"
base_model_name = "deb_quantile"

# Load command line arguments
parser = ArgumentParser()
parser.add_argument("pos_vars", nargs="*")  # gt_id and horizon
parser.add_argument(
    "--target_dates", "-t", default="std_paper_forecast", help="Specify which target dates to produce forecasts for"
)
parser.add_argument(
    "--correction_intensity", "-ci", type=float, default=1, help="Fraction, in [0, 1], of impact of correction over forecast"
)
parser.add_argument(
    "--correction_type",
    "-ct",
    default="additive",
    choices=["additive", "multiplicative"],
    help="Whether correction should be added (while ensuring no negative values) or multiplied",
)
args, opt = parser.parse_known_args()

# Add forecast argument
args.forecast_model = forecast

# Assign variables
gt_id = args.pos_vars[0]  # "contest_precip" or "contest_tmp2m"
horizon = args.pos_vars[1]  # "34w" or "56w"

# Reconstruct command-line arguments and run base model
cmd_args = " ".join(map(shlex.quote, sys.argv[1:]))
predict_script = resource_filename("subseasonal_toolkit", os.path.join("models", base_model_name, "batch_predict.py"))
cmd = f"python {predict_script} {cmd_args} -f {args.forecast_model}"

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
