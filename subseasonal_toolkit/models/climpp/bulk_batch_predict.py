# Generates predictions and metrics for each of the Climatology++ parameter configurations
# used by the tuner
#
# Example usage:
#   python -m subseasonal_toolkit.models.climpp.bulk_batch_predict contest_tmp2m 34w -t std_contest
#   python -m subseasonal_toolkit.models.climpp.bulk_batch_predict global_tmp2m_p1_1.5x1.5 34w -t s2s_eval -mo
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
#   --metrics_only (-mo): only generate metrics not predictions
import os
import subprocess
import shlex
from argparse import ArgumentParser
from pkg_resources import resource_filename
from subseasonal_toolkit.utils.general_util import printf
from subseasonal_toolkit.utils.models_util import get_submodel_name
from subseasonal_toolkit.utils.eval_util import get_named_targets

MODEL_NAME = "climpp"

# Parse command-line arguments
parser = ArgumentParser()
parser.add_argument("pos_vars",nargs="*")  # gt_id and target_horizon
parser.add_argument('--target_dates', '-t', default="std_test")
parser.add_argument('--cmd_prefix', '-c', default="python")
parser.add_argument('--metrics_only', '-mo', default=False, action='store_true')

args = parser.parse_args()
gt_id = args.pos_vars[0]
horizon = args.pos_vars[1]
target_dates = args.target_dates
cmd_prefix = args.cmd_prefix
metrics_only = args.metrics_only

# Specify list of parameter settings to run
if "tmp2m" in gt_id:
    metric = "mse" if ("_p1" in gt_id) or ("_p3" in gt_id) else "rmse"
    param_strs = [
        f"-l {metric} -y all -m 10", 
        f"-l {metric} -y all -m 7",
        f"-l {metric} -y all -m 1",
        f"-l {metric} -y all -m 0",
        f"-l {metric} -y 29 -m 10",
        f"-l {metric} -y 29 -m 7",
        f"-l {metric} -y 29 -m 1",
        f"-l {metric} -y 29 -m 0"
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

def param_str_to_submodel_name(param_str):
    """Returns submodel name for a given string of command line arguments for 
    batch_predict.py

    Args:
      param_str: string of command line arguments for batch_predict.py
    """
    # Set up argument parser to extract command-line arguments
    parser = ArgumentParser()
    # Loss used to learn parameters
    parser.add_argument('--loss', '-l', default="rmse", 
                        help="loss function: mse, rmse, skill, or ssm")
    # Number of years to use in training ("all" or integer)
    parser.add_argument('--num_years', '-y', default="all")
    # Number of month-day combinations on either side of the target combination to include
    # Set to 0 to include only target month-day combo
    # Set to 182 to include entire year
    parser.add_argument('--margin_in_days', '-m', default=0)
    parser.add_argument('--mei', default=False, action='store_true', help="Whether to condition on MEI")
    parser.add_argument('--mjo', default=False, action='store_true', help="Whether to condition on MJO")
    args, opt = parser.parse_known_args(shlex.split(param_str))
    
    # Assign variables
    loss = args.loss
    num_years = args.num_years
    mei = args.mei
    mjo = args.mjo
    if num_years != "all":
        num_years = int(num_years)
    margin_in_days = int(args.margin_in_days)

    return get_submodel_name(MODEL_NAME, loss=loss, num_years=num_years, 
                             margin_in_days=margin_in_days, mei=mei, mjo=mjo)

module_str = f"-m subseasonal_toolkit.models.{MODEL_NAME}.batch_predict"
task_str = f"{gt_id} {horizon} -t {target_dates}"

if cmd_prefix == "python":
    cmd_suffix = ""
else:
    # Add command suffix to get job dependency for metrics
    cmd_suffix = "| tail -n 1 | awk '{print $NF}'"
    # Include quotes for batch invocation
    module_str = f"\"{module_str}\""
        
# Set parameters for generating metrics files
metrics_prefix = cmd_prefix
metrics_script = resource_filename(__name__,os.path.join("..","..","batch_metrics.py"))
# Use wtd_mse metric for s2s gt_ids
if gt_id in ['global_tmp2m_p1_1.5x1.5', 'global_precip_p1_1.5x1.5',
             'global_tmp2m_p3_1.5x1.5', 'global_precip_p3_1.5x1.5']:
    metrics = "wtd_mse" 
else:
    metrics = "rmse score skill lat_lon_rmse"

for param_str in param_strs:
    #
    # Generate submodel predictions
    #
    if metrics_only:
        model_dependency=""
    else:
        cmd = f"{cmd_prefix} {module_str} {task_str} {param_str} {cmd_suffix}"
        printf(f"\nGenerate submodel predictions\n{cmd}")
        if cmd_prefix == "python":
            subprocess.call(cmd, shell=True)
        else:
            '''
            Get job dependency for metric evaluation
            '''            
            process = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, universal_newlines=True, shell=True)
            job_id = process.stdout.rstrip()
            model_dependency=f"-d {job_id}"

    #
    # Generate metrics for this submodel
    #
    submodel_name = param_str_to_submodel_name(param_str)
    if cmd_prefix == "python":
        metrics_cmd = f"python {metrics_script} {gt_id} {horizon} -mn {MODEL_NAME} -sn {submodel_name} -t {target_dates} -m {metrics}"
        printf(f"\nGenerate submodel metrics\n{metrics_cmd}")
        subprocess.call(metrics_cmd, shell=True)
    else:  
        '''
        Run dependent job for metric generation on named target date ranges
        '''
        if target_dates in get_named_targets():
            metrics_cmd=f"{metrics_prefix} {model_dependency} {metrics_script} {gt_id} {horizon} -mn {MODEL_NAME} -sn {submodel_name} -t {target_dates} -m {metrics}"
            printf(f"\nGenerate submodel metrics\n{metrics_cmd}")
            process = subprocess.run(metrics_cmd, shell=True)
