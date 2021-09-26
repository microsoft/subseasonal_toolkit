# Generates predictions for each of the Climatology++ parameter configurations
# used by the tuner
#
# Example usage:
#   python -m subseasonal_toolkit.models.climpp.bulk_batch_predict contest_tmp2m 34w -t std_contest
#       
# Positional args:
#   gt_id: contest_tmp2m, contest_precip, us_tmp2m, or us_precip
#   horizon: 34w or 56w
#
# Named args:
#   --target_dates (-t): target dates for batch prediction
#   --cmd_prefix (-c): prefix of command used to execute batch_predict.py
#     (default: "python"); e.g., "python" to run locally,
#     "src/batch/batch_python.sh --memory 12 --cores 16 --hours 1" to
#     submit to batch queue
import os
import subprocess
from argparse import ArgumentParser
from subseasonal_toolkit.utils.general_util import printf

model_name = "climpp"

# Parse command-line arguments
parser = ArgumentParser()
parser.add_argument("pos_vars",nargs="*")  # gt_id and target_horizon
parser.add_argument('--target_dates', '-t', default="std_test")
parser.add_argument('--cmd_prefix', '-c', default="python")

args = parser.parse_args()
gt_id = args.pos_vars[0]
horizon = args.pos_vars[1]
target_dates = args.target_dates
cmd_prefix = args.cmd_prefix

# Specify list of parameter settings to run
if "tmp2m" in gt_id:
    param_strs = [
        "-l rmse -y all -m 10", 
        "-l rmse -y all -m 7",
        "-l rmse -y all -m 1",
        "-l rmse -y all -m 0",
        "-l rmse -y 29 -m 10",
        "-l rmse -y 29 -m 7",
        "-l rmse -y 29 -m 1",
        "-l rmse -y 29 -m 0"
    ]
elif "precip" in gt_id:
    param_strs = [
        "-l mse -y all -m 10",
        "-l mse -y all -m 7",
        "-l mse -y all -m 1",
        "-l mse -y all -m 0"
    ]
else:
    raise ValueError(f"unknown gt_id {gt_id}")

module_str = f"-m subseasonal_toolkit.models.{model_name}.batch_predict"
task_str = f"{gt_id} {horizon} -t {target_dates}"
for param_str in param_strs:
    cmd = f"{cmd_prefix} \"{module_str}\" {task_str} {param_str}"
    printf(f"Running {cmd}")
    subprocess.call(cmd, shell=True)
