"""Generate batch predictions for localboosting's submodels for a given task and target dates. 

Example:
        $ python src/models/localboosting/bulk_batch_predict.py contest_tmp2m 34w -t std_paper 
        $ python src/models/localboosting/bulk_batch_predict.py us_precip 56w -t std_paper -c python
        $ python src/models/localboosting/bulk_batch_predict.py contest_tmp2m 34w -t std_contest -c "src/batch/batch_python.sh --memory 12 --cores 16 --hours 1"
        
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
from src.utils.general_util import printf  # general utility functions
from src.utils.eval_util import get_named_targets
from src.models.localboosting.attributes import MODEL_NAME



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

param_strs = [# ## selected
                "-re 2 -nf 10 -m 56 -i 50 -d 2 -lr 0.17",
                "-re 3 -nf 10 -m 56 -i 50 -d 2 -lr 0.17",
                "-re 2 -nf 20 -m 56 -i 50 -d 2 -lr 0.17",
                "-re 3 -nf 20 -m 56 -i 50 -d 2 -lr 0.17"]
submodel_names = [# ## selected
                    "localboosting-re_2-feat_10-m_56-iter_50-depth_2-lr_0_17",
                    "localboosting-re_3-feat_10-m_56-iter_50-depth_2-lr_0_17",
                    "localboosting-re_2-feat_20-m_56-iter_50-depth_2-lr_0_17",
                    "localboosting-re_3-feat_20-m_56-iter_50-depth_2-lr_0_17"]


    


cmd_script = os.path.join("src", "models", MODEL_NAME, "batch_predict.py")

# Add command suffix to get job dependency for metrics
if cmd_prefix == "python":
    cmd_suffix = ""
else:
    cmd_suffix = "| tail -n 1 | awk '{print $NF}'"
        
# Default parameters for generating metrics files
metrics_prefix="src/batch/batch_python.sh --memory 2 --cores 1 --hours 0 --minutes 3"
metrics_script="src/eval/batch_metrics.py"

for param_str, submodel_name in zip(param_strs, submodel_names):    
    '''
    Run submodel
    '''    
    cmd = f"{cmd_prefix} {cmd_script} {gt_id} {horizon} -t {target_date_str} {param_str} {cmd_suffix}"
    printf(f"Running {cmd}")
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



