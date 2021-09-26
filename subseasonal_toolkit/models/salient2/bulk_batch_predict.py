"""Generate batch predictions for salient2's submodels for a given task and target dates. 

Example:
        $ python src/models/salient2/bulk_batch_predict.py contest_tmp2m 34w -sn d2wk_cop_sst d2wk_cop_sst_mei -t std_paper
        $ python src/models/salient2/bulk_batch_predict.py contest_tmp2m 34w -sn d2wk_cop_sst d2wk_cop_sst_mei d2wk_cop_sst_mjo d2wk_cop_sst_mei_mjo -t std_paper
        
Positional args:
    gt_id: contest_tmp2m, contest_precip, us_tmp2m, or us_precip
    horizon: 34w or 56w

Named args:
    
    --target_dates (-t): target dates for batch prediction.
    --cmd_prefix (-c): prefix of command used to execute batch_predict.py
        (e.g., "python" to run locally,
        "src/batch/batch_python.sh --memory 12 --cores 16 --hours 1" to
        submit to batch queue)

"""

import os
import subprocess
from argparse import ArgumentParser
from subseasonal_toolkit.utils.general_util import printf  # general utility functions
from subseasonal_toolkit.utils.eval_util import get_named_targets
from subseasonal_toolkit.models.salient2.attributes import MODEL_NAME
from pkg_resources import resource_filename



# Load command line arguments
parser = ArgumentParser()
parser.add_argument("pos_vars",nargs="*")  # gt_id and target_horizon
parser.add_argument('--target_date_str', '-t', default="std_test")
parser.add_argument('--cmd_prefix', '-c', default="python")



args = parser.parse_args()

# Assign variables
gt_id = args.pos_vars[0] # "contest_precip", "contest_tmp2m", "us_precip" or "us_tmp2m"
horizon = args.pos_vars[1] # "34w" or "56w"
target_date_str = args.target_date_str
cmd_prefix = args.cmd_prefix



# Default parameters for generating prediction files
param_strs = ["-sn d2wk_cop_sst", "-sn d2wk_cop_sst_mei", "-sn d2wk_cop_sst_mjo", "-sn d2wk_cop_sst_mei_mjo"]
submodel_names = ["d2wk_cop_sst", "d2wk_cop_sst_mei", "d2wk_cop_sst_mjo", "d2wk_cop_sst_mei_mjo"]


# Default batch predict script
cmd_script = resource_filename("subseasonal_toolkit",os.path.join("src", "models", MODEL_NAME, "batch_predict.py"))

# Add command suffix to get job dependency for metrics
if cmd_prefix == "python":
    cmd_suffix = ""
else:
    cmd_suffix = "| tail -n 1 | awk '{print $NF}'"
        
# Default parameters for generating metrics files
batch_script=resource_filename("subseasonal_toolkit", os.path.join("src", "batch", "batch_python.sh"))
metrics_prefix=f"{batch_script} --memory 2 --cores 1 --hours 0 --minutes 3"
metrics_script=resource_filename("subseasonal_toolkit", os.path.join("src", "eval", "batch_metrics.py"))

for param_str, submodel_name in zip(param_strs, submodel_names):    
    '''
    Run submodel
    '''    
    cmd = f"{cmd_prefix} {cmd_script} {gt_id} {horizon} -t {target_date_str} {param_str} {cmd_suffix}"
    if cmd_prefix == "python":
        subprocess.call(cmd, shell=True)
    else:
        '''
        Get job dependency for metric evaluation
        '''
        printf(f"\nRunning submodel \n{cmd}")
        process = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, universal_newlines=True, shell=True)
        job_id = process.stdout.rstrip()
        model_dependency=f"-d {job_id}"

    '''
    Run metrics
    '''
    if cmd_prefix == "python":
        cmd = f"python {metrics_script} {gt_id} {horizon} -mn {MODEL_NAME} -sn {submodel_name} -t {target_date_str} -m rmse score skill lat_lon_rmse"
        subprocess.call(cmd, shell=True)
    else:  
        '''
        Run dependent job for metric generation on named target_date ranges
        '''
        if target_date_str in get_named_targets():
            metrics_cmd=f"{metrics_prefix} {model_dependency} {metrics_script} {gt_id} {horizon} -mn {MODEL_NAME} -sn {submodel_name} -t {target_date_str} -m rmse score skill lat_lon_rmse"
            printf(f"\nRunning metrics \n{metrics_cmd}") 
            process = subprocess.run(metrics_cmd, shell=True)



