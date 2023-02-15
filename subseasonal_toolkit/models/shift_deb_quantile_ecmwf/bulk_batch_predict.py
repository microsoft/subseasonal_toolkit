# Generates predictions for each of the shift_deb_quantile_ecmwf parameter configurations
#
# Example usage:
#   python -m subseasonal_toolkit.models.shift_deb_quantile_ecmwf.bulk_batch_predict us_tmp2m_1.5x1.5 34w -t std_paper_forecast
#       
# Positional args:
#   gt_id: e.g., contest_tmp2m, contest_precip, us_tmp2m, us_precip, us_tmp2m_1.5x1.5, us_precip_1.5x1.5
#   horizon: e.g., 12w, 34w, or 56w
#
# Named args:
#   --target_dates (-t): target dates for batch prediction
#   --cmd_prefix (-c): prefix of command used to execute batch_predict.py
#     (default: "python"); e.g., "python" to run locally,
#     "src/batch/batch_python.sh --memory 8 --cores 1 --hours 1" to
#     submit to batch queue
import os
import subprocess
import time
from argparse import ArgumentParser
from subseasonal_toolkit.utils.general_util import printf

forecast = "ecmwf"
model_name = f"shift_deb_quantile_{forecast}"

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

# Consider control forecast and each individual perturbed forecast
versions = [f"p{ii}" for ii in range(1,51)]+["c"]

module_str = f"-m subseasonal_toolkit.models.{model_name}.batch_predict"
if cmd_prefix.strip() != "python":
    # Include quotes for batch invocation
    module_str = f"\"{module_str}\""
    
task_str = f"{gt_id} {horizon} -t {target_dates}"
for version in versions:
    # Run batch predict for this configuration
    param_str=f"-fw {version} -ci 1 -ct additive"
    cmd=f"{cmd_prefix} {module_str} {task_str} {param_str}"
    printf(f"Running {cmd}")
    subprocess.call(cmd, shell=True)
    # Pause briefly between runs
    time.sleep(.1)
