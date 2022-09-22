# Generates predictions for each of the GEPS++ parameter configurations
# used by the tuner
#
# Example usage:
#   python -m subseasonal_toolkit.models.gempp.bulk_batch_predict contest_tmp2m 34w -t std_contest
#       
# Positional args:
#   gt_id: contest_tmp2m, contest_precip, us_tmp2m, or us_precip
#   horizon: 34w or 56w
#
# Named args:
#   --target_dates (-t): target dates for batch prediction
#   --cmd_prefix (-c): prefix of command used to execute batch_predict.py
#     (default: "python"); e.g., "python" to run locally,
#     "src/batch/batch_python.sh --memory 8 --cores 1 --hours 1" to
#     submit to batch queue
import os
import subprocess
from argparse import ArgumentParser
from subseasonal_toolkit.utils.general_util import printf

forecast = "gem"
model_name = f"{forecast}pp"

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
first_day = 1
years = 12
intercept = True
last_days = [1, 7, 14, 28, 42]
margins = [0, 14, 28, 35]
# Specify parallel arrays of first and last leads
if horizon == "56w":
    first_leads = [29]
    last_leads = [29]
elif horizon == "34w":
    first_leads = [18, 0, 15, 15]
    last_leads = [18, 18, 18, 15]
elif horizon == "12w":
    # No tuning for 12w lead
    first_leads = [1]
    last_leads = [1]
else:
    raise ValueError(f"invalid horizon {horizon}")
    
module_str = f"-m subseasonal_toolkit.models.{model_name}.batch_predict"
if cmd_prefix != "python":
    # Include quotes for batch invocation
    module_str = f"\"{module_str}\""

task_str = f"{gt_id} {horizon} -t {target_dates}"
# Iterate over parallel leads arrays
for ii in range(len(first_leads)):
    first_lead = first_leads[ii]
    last_lead = last_leads[ii]
    for last_day in last_days:
        for margin in margins:
            # Run batch predict for this configuration
            param_str=f"--forecast {forecast} -y {years} -m {margin} -i {intercept} -fd {first_day} -ld {last_day} -fl {first_lead} -ll {last_lead}"
            cmd=f"{cmd_prefix} {module_str} {task_str} {param_str}"
            printf(f"Running {cmd}")
            subprocess.call(cmd, shell=True)
