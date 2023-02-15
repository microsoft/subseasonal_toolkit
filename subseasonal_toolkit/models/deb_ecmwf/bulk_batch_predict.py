# Generates predictions for each of the ECMWF parameter configurations
# used by the tuner
#
# Example usage:
#   python -m subseasonal_toolkit.models.deb_ecmwf.bulk_batch_predict us_tmp2m_1.5x1.5 34w -t std_paper_forecast
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
import time
from argparse import ArgumentParser
from subseasonal_toolkit.utils.general_util import printf

model_name = "deb_ecmwf"

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
train_years = 20
loss = "mse"
# Specify parallel arrays of first and last leads
if horizon == "56w":
    first_leads = [29]
    last_leads = [29]
elif horizon == "34w":
    first_leads = [15]
    last_leads = [15]
elif horizon == "12w":
    first_leads = [1]
    last_leads = [1]
else:
    raise ValueError(f"invalid horizon {horizon}")
forecast_with = [f"p{ii}" for ii in range(1,51)] + ["c"]
debias_with = ["p+c"]
    
module_str = f"-m subseasonal_toolkit.models.{model_name}.batch_predict"
if cmd_prefix.strip() != "python":
    # Include quotes for batch invocation
    module_str = f"\"{module_str}\""
    
task_str = f"{gt_id} {horizon} -t {target_dates}"
# Iterate over parallel leads arrays
for ii in range(len(first_leads)):
    first_lead = first_leads[ii]
    last_lead = last_leads[ii]

    for fw in forecast_with:
        for dw in debias_with:
            # Run batch predict for this configuration
            param_str=f"-y {train_years} -l {loss} -fl {first_lead} -ll {last_lead} -fw {fw} -dw {dw}"
            cmd=f"{cmd_prefix} {module_str} {task_str} {param_str}"
            printf(f"Running {cmd}")
            subprocess.call(cmd, shell=True)
            # Pause briefly between runs
            time.sleep(.1)
